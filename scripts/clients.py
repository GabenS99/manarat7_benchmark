import os
import openai
import google.generativeai as genai
from groq import Groq
from mistralai import Mistral
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict

load_dotenv(override=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
deep_seek_key = os.getenv("DEEPSEEK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
mistral_api_key = os.getenv("MISTRAIL_API_KEY")

# Hugging Face token for accessing gated repositories
huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")


# Add 300-second timeout to all clients to prevent indefinite hanging
client_openai = openai.OpenAI(api_key=openai.api_key, timeout=300.0) if openai.api_key else None
client_deepseek = openai.OpenAI(api_key=deep_seek_key, base_url="https://api.deepseek.com", timeout=300.0) if deep_seek_key else None
client_groq = Groq(api_key=groq_api_key, timeout=300.0) if groq_api_key else None
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    client_gemini = genai
else:
    client_gemini = None
client_mistral = Mistral(api_key=mistral_api_key, timeout_ms=300000) if mistral_api_key else None  # Mistral uses milliseconds

def get_fanar_credentials() -> Tuple[Optional[str], Optional[Dict]]:
    """Get Fanar API URL and headers. Returns (url, headers) or (None, None)."""
    key = os.getenv('FANAR_API_KEY')
    if not key:
        print("[WARNING] Fanar API key not found")
        return None, None
    
    return (
        "https://api.fanar.qa/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )

allam_tokenizer = None
allam_model = None

jais2_tokenizer = None
jais2_model = None
jais2_loading_mode = None  # Track loading mode: "ONLINE", "OFFLINE", or None

