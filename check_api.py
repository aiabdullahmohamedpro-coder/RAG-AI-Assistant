
import os
from pathlib import Path
from dotenv import load_dotenv

project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key.strip():
    print("SUCCESS: API key loaded.")
else:
    print("ERROR: API key not found. Check your .env file.")
