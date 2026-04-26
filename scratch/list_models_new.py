import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print(f"[*] Checking models with NEW SDK for key: {api_key[:10]}...")

try:
    # In new SDK, we iterate through models
    for m in client.models.list():
        print(f" - {m.name} (Supports: {m.supported_actions})")
except Exception as e:
    print(f"\n[!] Error listing models: {e}")
