# OpenRouter API
OPENROUTER_API_KEY = "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"     
OPENROUTER_HTTP_REFERER = "http://localhost:8000"
OPENROUTER_X_TITLE = "Vindio AI Chatbot"
OPENROUTER_MODEL = "openai/gpt-oss-20b"

# Embeddings (HF) 
HF_EMBEDDING_MODEL="MahradHosseini/bge-m3-onnx-int8 "                 

# Index & data paths 
INDEX_DIR= "./index"
PAPER_PATH="./data/Paper.json"
QA_PATH="./data/QA.json"
WEB_PATH="./data/Web.json"

# Database (Supabase Postgres via psycopg3) 
# Supabase password: lVYDcNY3qIJyaTRq                             
# Supabase (Transaction Pooler)
DATABASE_URL = "postgresql+psycopg://postgres.emodmxawiibmaiokjxbn:lVYDcNY3qIJyaTRq@aws-1-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"

# Email
EMAIL_PROVIDER = "sendgrid"      
SEND_EMAILS = True               # master flag
SENDGRID_API_KEY = "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"
EMAIL_FROM = "homazabihi4@gmail.com"             # dont change it this is the verified one
EMAIL_TO_SUMMARIES = ["e258796@metu.edu.tr"]     # recipients: you can change it to test
SENDGRID_TEMPLATE_ID = "d-12eb1bf091b544e9b491b5786638d276"

# Scheduling
TIMEZONE = "Europe/Istanbul"
DAILY_DIGEST_HOUR = 9
WEEKLY_DIGEST_DAY = "mon"  # "mon".."sun"
WEEKLY_DIGEST_HOUR = 9

# Retention
RETENTION_MESSAGES_DAYS = 30
RETENTION_SUMMARIES_DAYS = 365
