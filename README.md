<!-- PROJECT HEADER -->
<div align="center">
  <h1>Vindio AI Software Ltd. Chatbot</h1>
  <p>
    <em>Domain-grounded AI assistant for Vindio AI Software Ltd</em><br>
    <em>Advanced Driver Assistance Systems (ADAS) and Quantum Vision research</em>
  </p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
    <img src="https://img.shields.io/badge/FastAPI-0.110+-green.svg" />
    <img src="https://img.shields.io/badge/HuggingFace-Optimum-yellow" />
    <img src="https://img.shields.io/badge/OpenRouter-Enabled-orange" />
    <img src="https://img.shields.io/badge/Supabase-Postgres-lightgreen" />
    <img src="https://img.shields.io/badge/License-AGPL--3.0-red" />
  </p>

  <!-- Demo GIF -->
  <img src="assets/demo.gif" alt="Vindio AI Chatbot Demo" width="80%" />
</div>

---

## 📖 Overview

The **Vindio AI Chatbot** is a high-performance, domain-specific AI assistant built for **Vindio AI Software Ltd**.  
It combines **Retrieval-Augmented Generation (RAG)** with multilingual embeddings, FAISS indexing, and OpenRouter LLM orchestration to deliver precise, grounded answers from:

- Internal Q&A knowledge base  
- The **Vindio AI Quantum Vision** research paper  
- Structured website snippets  

**Key Highlights:**
- ⚡ **Low-latency inference** with ONNX-optimized **BGE-M3** embeddings  
- 🔍 **FAISS vector store** for semantic similarity search  
- 🤖 **LLM pipeline** powered by **OpenRouter API**  
- 🌐 **FastAPI backend** + responsive static HTML/JS UI  
- 🗂️ **Session-based memory** with Postgres (Supabase) logging  
- 📧 Automated **daily/weekly email digests** with SendGrid  

---

## ✨ Features

- ✅ Retrieval-Augmented Generation (QA + Paper + Web)  
- ✅ Multilingual embeddings with **ONNX BGE-M3**  
- ✅ OpenRouter LLM integration  
- ✅ Configurable search (Top-K, min similarity, chunking)  
- ✅ FastAPI backend + modern static chat UI  
- ✅ CLI mode for quick testing  
- ✅ Supabase Postgres logging (users, sessions, messages, summaries)  
- ✅ Email summaries + automated digests via SendGrid  

---

## 🚀 Quick Start

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-org/vindio-ai-chatbot.git
cd vindio-ai-chatbot

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Configure environment
export OPENROUTER_API_KEY="sk-or-xxxx"
# Optional: configure Supabase DATABASE_URL and SendGrid API key in Config.py

# 4️⃣ Run FastAPI server
python Chatbot.py --serve --host 0.0.0.0 --port 8000

```
<div align="center"> <sub>© 2025 Vindio AI Software Ltd — Built by the Vindio AI LLM Team</sub> </div> ```
