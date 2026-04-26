import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print(f"[*] Checking models for key: {api_key[:10]}...")

try:
    models = genai.list_models()
    print("\n[Available Models]:")
    for m in models:
        print(f" - {m.name} (Supports: {m.supported_generation_methods})")
except Exception as e:
    print(f"\n[!] Error listing models: {e}")
