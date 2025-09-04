import os

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "Vindio AI Chatbot")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")

# Embeddings (HF) 
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "MahradHosseini/bge-m3-onnx-int8")

# Index & data paths 
INDEX_DIR = os.getenv("INDEX_DIR", "./index")
PAPER_PATH = os.getenv("PAPER_PATH", "./data/Paper.json")
QA_PATH = os.getenv("QA_PATH", "./data/QA.json")
WEB_PATH = os.getenv("WEB_PATH", "./data/Web.json")

# Database (Supabase Postgres via psycopg3) 
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Email
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "sendgrid")
SEND_EMAILS = os.getenv("SEND_EMAILS", "True").lower() == "true"
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_TO_SUMMARIES = os.getenv("EMAIL_TO_SUMMARIES", "").split(",") if os.getenv("EMAIL_TO_SUMMARIES") else []
SENDGRID_TEMPLATE_ID = os.getenv("SENDGRID_TEMPLATE_ID", "")

# Scheduling
TIMEZONE = os.getenv("TIMEZONE", "Europe/Istanbul")
DAILY_DIGEST_HOUR = int(os.getenv("DAILY_DIGEST_HOUR", "9"))
WEEKLY_DIGEST_DAY = os.getenv("WEEKLY_DIGEST_DAY", "mon")
WEEKLY_DIGEST_HOUR = int(os.getenv("WEEKLY_DIGEST_HOUR", "9"))

# Retention
RETENTION_MESSAGES_DAYS = int(os.getenv("RETENTION_MESSAGES_DAYS", "30"))
RETENTION_SUMMARIES_DAYS = int(os.getenv("RETENTION_SUMMARIES_DAYS", "365"))
