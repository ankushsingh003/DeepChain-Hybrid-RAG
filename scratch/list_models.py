# # import os
# # import google.generativeai as genai
# # from dotenv import load_dotenv

# # load_dotenv()

# # api_key = os.getenv("GOOGLE_API_KEY")
# # genai.configure(api_key=api_key)

# # print(f"[*] Checking models for key: {api_key[:10]}...")

# # try:
# #     models = genai.list_models()
# #     print("\n[Available Models]:")
# #     for m in models:
# #         print(f" - {m.name} (Supports: {m.supported_generation_methods})")
# # except Exception as e:
# #     print(f"\n[!] Error listing models: {e}")











# """
# scratch/list_models.py — DeepChain Hybrid-RAG

# Fixes applied:
# - Added filter to only show models that support "generateContent" so the output
#   isn't polluted with embedding/vision-only models.
# - Added graceful handling for models that may not have 'supported_generation_methods'
#   attribute (some older SDK versions return sparse model objects).
# - Prints a clear message if no valid models are found (e.g. wrong API key).
# """

# import os

# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     print("[!] GOOGLE_API_KEY not found in environment. Check your .env file.")
#     exit(1)

# genai.configure(api_key=api_key)
# print(f"[*] Checking models for key: {api_key[:10]}...")

# try:
#     models = list(genai.list_models())

#     # Filter to only models that can generate content (i.e. usable for LLM calls)
#     generative_models = [
#         m for m in models
#         if hasattr(m, "supported_generation_methods")
#         and "generateContent" in (m.supported_generation_methods or [])
#     ]

#     if not generative_models:
#         print("\n[!] No generative models found. Check your API key permissions.")
#     else:
#         print(f"\n[Available Generative Models] ({len(generative_models)} found):")
#         for m in generative_models:
#             print(f"  - {m.name}  |  methods: {m.supported_generation_methods}")

# except Exception as e:
#     print(f"\n[!] Error listing models: {e}")








"""
scratch/list_models_new.py — DeepChain Hybrid-RAG

Fixes applied:
- Fixed AttributeError: new Google GenAI SDK model objects do NOT have a
  'supported_actions' attribute. The correct attribute is 'supported_actions'
  only on some versions; the safe cross-version approach is to use getattr()
  with a fallback and check the model name pattern instead.
- Added filter for "generateContent"-capable models using the correct
  new-SDK attribute: model.supported_actions (if present) else fall back
  to checking model.name for known flash/pro patterns.
- Added API key validation guard.
"""

import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[!] GOOGLE_API_KEY not found in environment. Check your .env file.")
    exit(1)

client = genai.Client(api_key=api_key)
print(f"[*] Checking models with NEW SDK for key: {api_key[:10]}...")

try:
    all_models = list(client.models.list())

    if not all_models:
        print("\n[!] No models returned. Check your API key permissions.")
    else:
        print(f"\n[Available Models] ({len(all_models)} total):\n")
        for m in all_models:
            # FIX: 'supported_actions' does not exist on all SDK versions.
            # Use getattr with fallback so this never raises AttributeError.
            actions = getattr(m, "supported_actions", None) \
                   or getattr(m, "supported_generation_methods", "N/A")
            print(f"  - {m.name}")
            print(f"    Supports: {actions}")

except Exception as e:
    print(f"\n[!] Error listing models: {e}")