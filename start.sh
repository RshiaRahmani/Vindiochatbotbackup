#!/bin/bash
# Start script for Render deployment
python Chatbot.py --serve --host 0.0.0.0 --port ${PORT:-10000}
