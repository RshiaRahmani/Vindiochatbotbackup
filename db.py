# --------------------------------------------------------------------------------------
# db.py — Database Helpers for Vindio AI Chatbot:
# • Provides a thin SQLAlchemy wrapper for Postgres access.
# • Manages persistence of users, sessions, messages, and conversation summaries.
# • Handles write-only operations (DATABASE_URL is read from Config.py).
# --------------------------------------------------------------------------------------
from typing import Optional
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from Config import DATABASE_URL  

if not DATABASE_URL or not isinstance(DATABASE_URL, str):
    raise RuntimeError("Config.DATABASE_URL must be a non-empty string.")

# Connection pool tuned for a small FastAPI service
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    pool_recycle=300,  # prevent stale connections
    future=True,       # SQLAlchemy 2.0 style
)

# ----------------------------------------------------------------------------
# Core write helpers
# ----------------------------------------------------------------------------
def get_or_create_user(ip: str):
    """
    Returns the user's UUID. Creates a new row if this IP hasn't been seen before.
    """
    if not ip:
        ip = "0.0.0.0"
    with engine.begin() as cx:
        row = cx.execute(
            text("SELECT id FROM users WHERE ip = :ip ORDER BY created_at ASC LIMIT 1"),
            {"ip": ip},
        ).fetchone()
        if row:
            return row[0]
        return cx.execute(
            text("INSERT INTO users (ip) VALUES (:ip) RETURNING id"),
            {"ip": ip},
        ).scalar_one()

def get_or_create_session(user_id: str, sid: Optional[str], ua: Optional[str], ref: Optional[str]):
    """
    Returns the session UUID. If 'sid' exists in DB, re-use it; otherwise creates a new session row.
    Note: DB generates the UUID; we do not force it to equal any cookie value.
    """
    with engine.begin() as cx:
        if sid:
            row = cx.execute(
                text("SELECT id FROM sessions WHERE id = :sid"),
                {"sid": sid},
            ).fetchone()
            if row:
                return row[0]
        return cx.execute(
            text(
                """
                INSERT INTO sessions (user_id, user_agent, referrer)
                VALUES (:user_id, :ua, :ref)
                RETURNING id
                """
            ),
            {"user_id": user_id, "ua": ua, "ref": ref},
        ).scalar_one()

def log_message(
    session_id: str,
    role: str,
    content: str,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    latency_ms: Optional[int] = None,
    metadata: Optional[dict] = None,
):
    """
    Inserts a single message row (user/assistant/system).
    Uses JSONB bind param so dicts are correctly stored in Postgres.
    """
    stmt = (
        text(
            """
            INSERT INTO messages
                (session_id, role, content, tokens_in, tokens_out, latency_ms, metadata)
            VALUES
                (:sid, :role, :content, :tin, :tout, :lat, :meta)
            """
        )
        .bindparams(bindparam("meta", type_=JSONB))
    )
    with engine.begin() as cx:
        cx.execute(
            stmt,
            {
                "sid": session_id,
                "role": role,
                "content": content,
                "tin": tokens_in,
                "tout": tokens_out,
                "lat": latency_ms,
                "meta": metadata,
            },
        )

def end_session(session_id: str):
    """
    Marks a session as ended (sets ended_at = now()).
    """
    with engine.begin() as cx:
        cx.execute(
            text("UPDATE sessions SET ended_at = NOW() WHERE id = :sid AND ended_at IS NULL"),
            {"sid": session_id},
        )

def write_summary(
    session_id: str,
    summary: str,
    top_intent: Optional[str] = None,
    intents: Optional[list[str]] = None,
):
    """
    Upserts a session-level conversation summary.
    """
    stmt = (
        text(
            """
            INSERT INTO conversation_summaries (session_id, summary, top_intent, intents)
            VALUES (:sid, :sum, :top, :ints)
            ON CONFLICT (session_id) DO UPDATE
              SET summary = EXCLUDED.summary,
                  top_intent = EXCLUDED.top_intent,
                  intents = EXCLUDED.intents
            """
        )
    )
    with engine.begin() as cx:
        cx.execute(
            stmt,
            {"sid": session_id, "sum": summary, "top": top_intent, "ints": intents},
        )

def healthcheck():
    from sqlalchemy import text
    with engine.connect() as cx:
        return cx.execute(text("select version()")).scalar_one()
