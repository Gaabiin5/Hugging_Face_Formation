import requests

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
