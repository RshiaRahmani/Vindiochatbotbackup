# --------------------------------------------------------------------------------------
# RAG.py:
# Retrieval-Augmented Generation (retriever-only):
# - Loads Q&A, paper JSON, and optional web JSON
# - Splits long documents into overlapping word chunks
# - Embeds all text using EmbeddingModel (BGE-M3 ONNX)
# - Builds FAISS inner-product (cosine) indexes for fast similarity search
# - Caches indexes + mapping metadata under INDEX_DIR to skip rebuilds on restart
#
# Note: This class does NOT generate answers. It only retrieves relevant snippets.
# --------------------------------------------------------------------------------------
from __future__ import annotations
from typing import List, Optional, Any, Dict, Tuple
import os
import json
import numpy as np
import faiss
from EmbeddingModel import EmbeddingModel
import logging
import hashlib
import pickle
from pathlib import Path
from Config import INDEX_DIR

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Cache fingerprint helpers
# --------------------------------------------------------------------------------------
def _sha256_str(s: str) -> str:
    """Return hex SHA-256 for a string (used to key cache files)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _make_build_fingerprint(
    model_id: str,
    questions_path: str,
    answers_path: str,
    paper_path: str,
    web_path: str,
    paper_chunk_size: int,
    paper_chunk_overlap: int,
    enable_paper_chunking: bool,
    max_seq_len: int,
) -> str:
    """
    Build a stable fingerprint based on model & data inputs + key parameters.
    Any change here invalidates the cache and triggers a rebuild.
    """
    def stat(p: str) -> Tuple[float, int]:
        try:
            st = Path(p).stat()
            return (st.st_mtime, st.st_size)
        except FileNotFoundError:
            return (0.0, 0)

    q_m, q_s = stat(questions_path)
    a_m, a_s = stat(answers_path) if answers_path else (0.0, 0)
    p_m, p_s = stat(paper_path)
    w_m, w_s = stat(web_path) if web_path else (0.0, 0)

    meta = {
        "rag_ver": "3",  
        "model_id": (model_id or "").strip(),
        "max_seq_len": int(max_seq_len),
        "questions_path": str(Path(questions_path)),
        "answers_path": str(Path(answers_path)) if answers_path else "",
        "paper_path": str(Path(paper_path)),
        "web_path": str(Path(web_path)) if web_path else "",
        "q_mtime": q_m,
        "q_size": q_s,
        "a_mtime": a_m,
        "a_size": a_s,
        "p_mtime": p_m,
        "p_size": p_s,
        "w_mtime": w_m,
        "w_size": w_s,
        "chunk_size": int(paper_chunk_size),
        "chunk_overlap": int(paper_chunk_overlap),
        "chunking": bool(enable_paper_chunking),
    }
    return _sha256_str(json.dumps(meta, sort_keys=True))


def _cache_paths(index_dir: str, fp: str) -> Dict[str, str]:
    """Return file locations for FAISS + metadata based on fingerprint."""
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "qa_union_idx": str(d / f"qa_union_{fp}.faiss"),
        "paper_idx":    str(d / f"paper_{fp}.faiss"),
        "web_idx":      str(d / f"web_{fp}.faiss"),
        "maps":         str(d / f"maps_{fp}.pkl"),
    }

# --------------------------------------------------------------------------------------
# RAG class
# --------------------------------------------------------------------------------------
class RAG:
    """
    Pipeline:
      load → split → embed → index (Q, A, Q∪A, Paper, Web)

    Public surface:
      - search_structured(query, ...) -> list[dict]
        Returns typed hits with metadata for QA / paper / web.
    """
    def __init__(
        self,
        model_id: str,
        questions_path: str,
        answers_path: str,
        paper_path: str,
        web_path: Optional[str] = None,
        onnx_file_name: Optional[str] = "model_quantized.onnx",
        max_seq_len: int = 8192,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        # Word-chunking settings (applied to paper & web content)
        paper_chunk_size: int = 160,           
        paper_chunk_overlap: int = 40,        
        enable_paper_chunking: bool = True,  
        # Output limits for 'full_content' fields returned to LLM/UI  
        full_content_char_limit: Optional[int] = None,  
        # Always store/search normalized vectors for IP (cosine) search
        force_normalize_for_ip: bool = True,           
    ) -> None:
        logger.info("Initializing RAG retriever...")
        

        # ------------------------
        # 1) Load JSON data
        # ------------------------
        logger.info(f"Loading questions from {questions_path}")
        self.qa_items: List[Dict[str, Any]] = self._load_qa(questions_path)

        # Optional second QA file (e.g., answers merged with questions)
        if answers_path and os.path.abspath(answers_path) != os.path.abspath(questions_path):
            logger.info(f"Loading answers from {answers_path}")
            self.qa_items.extend(self._load_qa(answers_path))

        logger.info(f"Loading paper content from {paper_path}")    
        self.paper_items: List[Dict[str, Any]] = self._load_paper(paper_path)

        self.web_items: List[Dict[str, Any]] = []
        self.web_enabled: bool = bool(web_path)
        self.web_path = web_path or ""
        if self.web_enabled:
            logger.info(f"Loading web content from {self.web_path}")
            self.web_items = self._load_web(self.web_path)

        # ------------------------
        # 2) Prepare texts to embed
        #    - QA is kept as separate lists (Q and A) and later deduped on union hits
        #    - Paper/Web are word-chunked (with overlap) for better recall
        # ------------------------
        logger.debug("Preparing question and answer text lists...")
        self.q_idx_map: List[int] = []  # union-row (Q block) -> qa_items index
        self.a_idx_map: List[int] = []  # union-row (A block) -> qa_items index
        self.q_texts: List[str] = []
        self.a_texts: List[str] = []

        for i, it in enumerate(self.qa_items):
            q = str(it.get("Question", "")).strip()
            a = str(it.get("Answer", "")).strip()
            if q:
                self.q_idx_map.append(i)
                self.q_texts.append(q)
            if a:
                self.a_idx_map.append(i)
                self.a_texts.append(a)

        # Paper chunks
        logger.debug("Preparing paper text chunks...")
        self.p_idx_map: List[int] = []      # faiss row -> original paper_items index
        self.p_chunk_ids: List[int] = []    # faiss row -> chunk id within that item
        self.p_texts: List[str] = []
        for i, it in enumerate(self.paper_items):
            c = str(it.get("content", "")).strip()
            if not c:
                continue
            if enable_paper_chunking and paper_chunk_size > 0:
                chunks = self._split_into_word_chunks(
                    c, chunk_size=paper_chunk_size, overlap=paper_chunk_overlap
                )
                for j, ch in enumerate(chunks):
                    self.p_idx_map.append(i)
                    self.p_chunk_ids.append(j)
                    self.p_texts.append(ch)
            else:
                # No chunking: one row per item
                self.p_idx_map.append(i)
                self.p_chunk_ids.append(0)
                self.p_texts.append(c)

        # Web chunks (same policy as paper)
        logger.debug("Preparing web text chunks...")
        self.w_idx_map: List[int] = []      # faiss row -> original web_items index
        self.w_chunk_ids: List[int] = []    # faiss row -> chunk id within that item
        self.w_texts: List[str] = []
        if self.web_enabled:
            for i, it in enumerate(self.web_items):
                c = str(it.get("content", "")).strip()
                if not c:
                    continue
                if enable_paper_chunking and paper_chunk_size > 0:
                    chunks = self._split_into_word_chunks(
                        c, chunk_size=paper_chunk_size, overlap=paper_chunk_overlap
                    )
                    for j, ch in enumerate(chunks):
                        self.w_idx_map.append(i)
                        self.w_chunk_ids.append(j)
                        self.w_texts.append(ch)
                else:
                    self.w_idx_map.append(i)
                    self.w_chunk_ids.append(0)
                    self.w_texts.append(c)

        # ------------------------
        # 3) Embedding model
        # ------------------------
        logger.info("Loading embedding model...")
        self.embedding_model = EmbeddingModel(
            model_id=model_id,
            onnx_file_name=onnx_file_name,
            max_seq_len=max_seq_len,
            normalize=normalize,   
            cache_dir=cache_dir,
        )

        # class-level options
        self.full_content_char_limit: Optional[int] = full_content_char_limit
        self.force_normalize_for_ip: bool = bool(force_normalize_for_ip)

        # ------------------------
        # 3.5) Try loading from disk cache
        # ------------------------
        fp = _make_build_fingerprint(
            model_id=model_id,
            questions_path=questions_path,
            answers_path=answers_path or "",
            paper_path=paper_path,
            paper_chunk_size=paper_chunk_size,
            paper_chunk_overlap=paper_chunk_overlap,
            enable_paper_chunking=enable_paper_chunking,
            max_seq_len=max_seq_len,
            web_path=self.web_path,
        )
        paths = _cache_paths(INDEX_DIR, fp)

        # Placeholders for FAISS indexes
        self.q_index = None
        self.a_index = None
        self.qa_union_index = None
        self.p_index = None
        self.w_index = None
        self._qa_union_split = (0, 0)  

        # If all required cache files exist, restore them
        if (
            os.path.exists(paths["qa_union_idx"]) and
            os.path.exists(paths["paper_idx"]) and
            os.path.exists(paths["maps"]) and
            ((not self.web_enabled) or os.path.exists(paths["web_idx"]))
        ):
            try:
                logger.info("Loading FAISS indexes from cache...")
                self.qa_union_index = faiss.read_index(paths["qa_union_idx"])
                self.p_index = faiss.read_index(paths["paper_idx"])
                if self.web_enabled:
                    self.w_index = faiss.read_index(paths["web_idx"])
                with open(paths["maps"], "rb") as f:
                    maps = pickle.load(f)
                
                # Restore mapping arrays
                self.q_idx_map = list(maps.get("q_idx_map", []))
                self.a_idx_map = list(maps.get("a_idx_map", []))
                self.p_idx_map = list(maps.get("p_idx_map", []))
                self.p_chunk_ids = list(maps.get("p_chunk_ids", []))
                self.w_idx_map     = list(maps.get("w_idx_map", []))
                self.w_chunk_ids   = list(maps.get("w_chunk_ids", []))
                self._qa_union_split = tuple(maps.get("qa_union_split", (len(self.q_idx_map), len(self.a_idx_map)))) 

                logger.info("RAG retriever initialized from cache.")
                return
            except Exception as e:
                logger.warning(f"Cache load failed; rebuilding indexes. Reason: {e}")

        # ------------------------
        # 4) Embed texts & build FAISS
        # ------------------------
        logger.info("Embedding Q/A/Paper texts...")
        self.q_emb = self._embed_or_empty(self.q_texts)
        self.a_emb = self._embed_or_empty(self.a_texts)
        self.p_emb = self._embed_or_empty(self.p_texts)
        self.w_emb = self._embed_or_empty(self.w_texts) if self.web_enabled else np.zeros((0,0), dtype=np.float32)

        dim = self._infer_dim([self.q_emb, self.a_emb, self.p_emb, self.w_emb])

        if dim is not None:
            logger.info(f"Building FAISS indexes with vector dimension {dim}...")
            if self.q_emb.size:
                self.q_index = self._build_ip_index(self.q_emb, dim)
            if self.a_emb.size:
                self.a_index = self._build_ip_index(self.a_emb, dim)
            if self.p_emb.size:
                self.p_index = self._build_ip_index(self.p_emb, dim)
            if self.web_enabled and self.w_emb.size:
                self.w_index = self._build_ip_index(self.w_emb, dim)

            # Union index (Questions ∪ Answers) for one-pass QA retrieval with deduping
            if self.q_emb.size or self.a_emb.size:
                blocks = []
                if self.q_emb.size:
                    blocks.append(self.q_emb)
                if self.a_emb.size:
                    blocks.append(self.a_emb)
                qa_union_emb = np.vstack(blocks) if blocks else np.zeros((0, dim), dtype=np.float32)
                if qa_union_emb.size:
                    self.qa_union_index = self._build_ip_index(qa_union_emb, dim)
                    self._qa_union_split = (self.q_emb.shape[0], self.a_emb.shape[0])
        
        # ------------------------
        # 4.5) Persist to cache (best-effort)
        # ------------------------
        try:
            logger.info("Saving FAISS indexes to cache...")
            if self.qa_union_index is not None:
                faiss.write_index(self.qa_union_index, paths["qa_union_idx"])
            if self.p_index is not None:
                faiss.write_index(self.p_index, paths["paper_idx"])
            if self.web_enabled and self.w_index is not None:
                faiss.write_index(self.w_index, paths["web_idx"])
            with open(paths["maps"], "wb") as f:
                pickle.dump({
                    "q_idx_map": self.q_idx_map,
                    "a_idx_map": self.a_idx_map,
                    "p_idx_map": self.p_idx_map,
                    "p_chunk_ids": self.p_chunk_ids,
                    "w_idx_map": self.w_idx_map,
                    "w_chunk_ids": self.w_chunk_ids,
                    "qa_union_split": self._qa_union_split,
                }, f)
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")
        
        logger.info("RAG retriever initialized successfully.")

    #-----------------------------------------------------------------------------------
    # Structured search: returns dicts with type/id/score/text and metadata
    #-----------------------------------------------------------------------------------
    def search_structured(
        self,
        query: str,
        top_k_qa: int = 5,
        top_k_paper: int = 5,
        top_k_web: int = 5,
        min_score: Optional[float] = None,
        return_full_content: bool = True,       
        full_content_char_limit: Optional[int] = None,  
    ) -> List[Dict[str, Any]]:
        """
        Search over QA, paper and optional web indexes (cosine similarity via IP).

        Returns list of dicts like:
          QA:
            {"type":"qa","id":int,"score":float,"text":str,"question":str,"answer":str}
          Paper:
            {"type":"paper","id":int,"score":float,"text":str,"snippet":str,
             "full_content":str,"page":str,"section":str,"chunk_id":int}
          Web:
            {"type":"web","id":int,"score":float,"text":str,"snippet":str,
             "full_content":str,"url":str,"section":str,"chunk_id":int}
        """
        # Guard: empty query
        if not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query string received.")
            return []
        
        logger.info(f"Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        qvec = self._prepare_query_vector(query)
        results: List[Dict[str, Any]] = []

        # ------------------------
        # (Q ∪ A) search with deduplication at QA-item level
        # ------------------------
        seen_pairs: set[int] = set()
        qa_results: List[Dict[str, Any]] = []
        nq, _ = self._qa_union_split

        if self.qa_union_index is not None and self.qa_union_index.ntotal > 0 and top_k_qa > 0:
            # Over-fetch to compensate for dedupe + min_score filtering
            raw_k = min(max(top_k_qa * 3, top_k_qa), self.qa_union_index.ntotal)
            D, I = self.qa_union_index.search(qvec, raw_k)
            logger.debug(f"Found {len(I[0])} QA candidates.")

            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0:
                    continue
                if min_score is not None and score < float(min_score):
                    continue

                # Map union-row index back to original qa_items index
                qa_row = self.q_idx_map[idx] if idx < nq else self.a_idx_map[idx - nq]
                if qa_row in seen_pairs:
                    continue
                seen_pairs.add(qa_row)

                item = self.qa_items[qa_row]
                q = str(item.get("Question", "")).strip()
                a = str(item.get("Answer", "")).strip()
                if q or a:
                    qa_results.append({
                        "type": "qa",
                        "id": int(qa_row),
                        "score": float(score),
                        "text": f"Q: {q}\nA: {a}",
                        "question": q,
                        "answer": a,
                    })
                    if len(qa_results) >= top_k_qa:
                        break

        results.extend(qa_results)

        # ------------------------
        # Web search 
        # ------------------------
        if self.web_enabled and self.w_index is not None and self.w_index.ntotal > 0 and top_k_web > 0:
            k = min(top_k_web, self.w_index.ntotal)
            D, I = self.w_index.search(qvec, k)

             # Determine how much 'full_content' to include (per-call override > default)
            local_limit = (
                full_content_char_limit
                if full_content_char_limit is not None
                else self.full_content_char_limit
            )

            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0:
                    continue
                if min_score is not None and score < float(min_score):
                    continue
                if idx >= len(self.w_idx_map):
                    continue

                doc_id = int(self.w_idx_map[idx])
                it = self.web_items[doc_id]
                url = str(it.get("url", "")).strip()
                sec = str(it.get("Section", "")).strip()
                chunk_id = self.w_chunk_ids[idx] if idx < len(self.w_chunk_ids) else 0

                snippet = self.w_texts[idx] if 0 <= idx < len(self.w_texts) else ""
                full_content = str(it.get("content", "")).strip()
                if isinstance(local_limit, int) and local_limit > 0 and len(full_content) > local_limit:
                    full_content = full_content[:local_limit]

                prefix = f"[{sec or 'WEB'} | chunk {chunk_id}]".strip()

                results.append({
                    "type": "web",
                    "id": doc_id,
                    "score": float(score),
                    "text": f"{prefix} {snippet}",
                    "snippet": snippet,
                    "full_content": full_content,
                    "url": url,
                    "section": sec,
                    "chunk_id": int(chunk_id),
                })

        # ------------------------
        # Paper search
        # ------------------------
        if self.p_index is not None and self.p_index.ntotal > 0 and top_k_paper > 0:
            k = min(top_k_paper, self.p_index.ntotal)
            D, I = self.p_index.search(qvec, k)
            logger.debug(f"Found {len(I[0])} paper snippet candidates.")

            # per-call override > class default
            local_limit = (
                full_content_char_limit
                if full_content_char_limit is not None
                else self.full_content_char_limit
            )

            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx < 0:
                    continue
                if min_score is not None and score < float(min_score):
                    continue
                if idx >= len(self.p_idx_map):
                    logger.warning(f"Paper index out of bounds: {idx}")
                    continue  # safety guard

                doc_id = int(self.p_idx_map[idx])
                it = self.paper_items[doc_id]
                page = str(it.get("Page_no", "")).strip()
                sec  = str(it.get("Section", "")).strip()
                chunk_id = self.p_chunk_ids[idx] if idx < len(self.p_chunk_ids) else 0

                # Use the exact retrieved chunk as the snippet
                snippet = self.p_texts[idx] if 0 <= idx < len(self.p_texts) else ""

                # Attach the full document content for the LLM
                full_content = str(it.get("content", "")).strip()
                if isinstance(local_limit, int) and local_limit > 0 and len(full_content) > local_limit:
                    full_content = full_content[:local_limit]

                prefix = f"[Page {page} | {sec} | chunk {chunk_id}]".strip()


                results.append({
                    "type": "paper",
                    "id": doc_id,
                    "score": float(score),
                    "text": f"{prefix} {snippet}",
                    "snippet": snippet,
                    "full_content": full_content,
                    "page": page,
                    "section": sec,
                    "chunk_id": int(chunk_id),
                })

        logger.info(f"Search complete. Returning {len(results)} results.")
        return results
    
    # ----------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------
    def _load_json_list(self, path: str) -> List[Any]:
        """Load a JSON file and assert it contains a list at the root."""
        if not os.path.exists(path):
            logger.error(f"JSON file not found: {path}")
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"{path} must contain a JSON list at root.")
            raise ValueError(f"{path} must be a JSON list at the root.")
        return data


    def _load_qa(self, path: str) -> List[Dict[str, Any]]:
        """QA loader: validates presence of 'Question' and 'Answer' keys."""
        logger.debug(f"Loading QA from {path}")
        data = self._load_json_list(path)
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(data):
            if not isinstance(it, dict):
                raise ValueError(f"QA item {i} must be an object, got {type(it)}")
            if "Question" not in it or "Answer" not in it:
                raise ValueError(
                    f"QA item {i} must include 'Question' and 'Answer' fields. Got keys: {list(it.keys())}"
                )
            out.append(it)
        return out


    def _load_paper(self, path: str) -> List[Dict[str, Any]]:
        """Paper loader: validates presence of 'content' key (plus optional metadata)."""
        logger.debug(f"Loading paper data from {path}")
        data = self._load_json_list(path)
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(data):
            if not isinstance(it, dict):
                raise ValueError(f"Paper item {i} must be an object, got {type(it)}")
            if "content" not in it:
                raise ValueError(
                    f"Paper item {i} must include 'content' field. Got keys: {list(it.keys())}"
                )
            out.append(it)
        return out
    

    def _load_web(self, path: str) -> List[Dict[str, Any]]:
        """Web loader: validates 'content'; ensures 'url' and 'Section' exist for formatting."""
        logger.debug(f"Loading web data from {path}")
        data = self._load_json_list(path)
        out: List[Dict[str, Any]] = []
        for i, it in enumerate(data):
            if not isinstance(it, dict):
                raise ValueError(f"Web item {i} must be an object, got {type(it)}")
            if "content" not in it:
                raise ValueError(
                    f"Web item {i} must include 'content' field. Got keys: {list(it.keys())}"
                )
            it.setdefault("url", "")
            it.setdefault("Section", "")
            out.append(it)
        return out
    

    def _embed_or_empty(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts or return an empty (0x0) float32 array if none."""
        if not texts:
            logger.debug("No texts to embed.")
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self.embedding_model.embed_multiple_documents(texts), dtype=np.float32)


    def _infer_dim(self, arrays: List[np.ndarray]) -> Optional[int]:
        """Infer embedding dimension from the first non-empty array in a list."""
        for a in arrays:
            if isinstance(a, np.ndarray) and a.size > 0:
                return int(a.shape[1])
        return None


    def _build_ip_index(self, vecs: np.ndarray, dim: int) -> faiss.Index:
        """Build a FAISS IndexFlatIP (inner product) index, expecting unit-norm vectors."""
        logger.debug(f"Building FAISS IP index with {vecs.shape[0]} vectors of dim {dim}.")
        if vecs.size == 0:
            raise ValueError("Cannot build index on empty vectors.")
        if vecs.shape[1] != dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {dim}, got {vecs.shape[1]}.")
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        return index
    

    def _prepare_query_vector(self, query: str) -> np.ndarray:
        """Embed a single query and return it shaped as (1, dim) float32."""
        logger.debug("Preparing query vector...")
        # EmbeddingModel already returns L2-normalized vectors when normalize=True.
        q = np.asarray(self.embedding_model.embed_single_document(query), dtype=np.float32)[None, :]
        return q


    def _split_into_word_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into word chunks with overlap"""
        words = text.split()
        if not words:
            return []
        if chunk_size <= 0:
            return [" ".join(words)]
        overlap = max(0, min(overlap, chunk_size - 1))
        step = chunk_size - overlap
        chunks = []
        for start in range(0, len(words), step):
            chunk = words[start:start + chunk_size]
            if not chunk:
                break
            chunks.append(" ".join(chunk))
            if start + chunk_size >= len(words):
                break
        return chunks
