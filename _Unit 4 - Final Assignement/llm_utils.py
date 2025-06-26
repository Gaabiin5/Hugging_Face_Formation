import requests
import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, GoogleAPICallError


def ollama_chat_completion(system: str, messages: list, model: str = "llama3", stream: bool = False) -> str:
    """
    Call a local Ollama model (e.g., llama3) with system + messages prompt.

    Returns the generated response content (string).
    """
    full_messages = [{"role": "system", "content": system}] + messages

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": full_messages,
            "stream": stream
        }
    )
    if not response.ok:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    return response.json()["message"]["content"]


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_chat_completion(system: str, messages: list, model: str = "gemini-pro") -> str:
    """
    Utilise Gemini (via Google Generative AI SDK) en mode chat complet.
    """
    chat = genai.GenerativeModel(model).start_chat(history=[])
    # Concatène prompt système et messages en un seul bloc d’initialisation
    prompt = f"{system}\n\n"
    for msg in messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    response = chat.send_message(prompt)
    return response.text.strip()

import time

def safe_gemini_chat_completion(system: str, messages: list, model: str = "models/gemini-1.5-flash", retries=3, wait_time=20) -> str:
    """
    Retry wrapper for Gemini API with automatic wait on quota errors or failures.
    """
    for attempt in range(1, retries + 1):
        try:
            return gemini_chat_completion(system, messages, model)
        
        except ResourceExhausted as e:
            print(f"⚠️ [Quota] Attempt {attempt}/{retries}: {e}")
            time.sleep(wait_time)

        except DeadlineExceeded as e:
            print(f"⏱️ [Timeout] Attempt {attempt}/{retries}: {e}")
            time.sleep(wait_time)

        except GoogleAPICallError as e:
            print(f"🌐 [API Error] Attempt {attempt}/{retries}: {e}")
            time.sleep(wait_time)

        except Exception as e:
            print(f"❌ [Unknown Error] Attempt {attempt}/{retries}: {e}")
            time.sleep(wait_time)

    raise RuntimeError("❗ Gemini API failed after maximum retries.")
