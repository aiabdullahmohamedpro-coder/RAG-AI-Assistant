
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load the API key from .env
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API key not found. Check your .env file.")

try:
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    # Send a small test request
    response = llm.invoke(
        "Reply with exactly: OpenAI connection successful."
    )

    print("SUCCESS: API request completed.")
    print("Model response:", response.content)

except Exception as error:
    print("API request failed.")
    print("Error type:", type(error).__name__)
    print("Error details:", str(error)[:500])
