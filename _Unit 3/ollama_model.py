# ollama_model.py

print("OLLAMA MODEL MODULE LOADED")


import requests

SYSTEM_PROMPT = (
    "You are an AI assistant executing tasks step-by-step using code.\n"
    "Always follow this structure:\n"
    "Thoughts: Reason about the next action\n"
    "Code:\n```py\n# your Python code here\n```\n<end_code>\n\n"
    "Only use the following tools:\n"
    "- def weather_info(location: str) -> str\n"
    "- def web_search(query: str) -> str\n"
    "- def guest_info_retriever(query: str) -> str\n"
    "- def final_answer(answer: any) -> any\n"
    "\n"
    "`weather_info` only takes one argument: a string.\n"
    "`guest_info_retriever` is the only way to access guest information. Provide either a guest name, a relation (e.g., 'best friend'), or 'all' to retrieve a full list.\n"
    "Never attempt to access files, open CSVs, or use 'open()', 'csv', or similar — these are forbidden.\n"
    "\n"
    "Avoid non-ASCII characters like `°`, use `deg` instead.\n"
    "Do not reference undefined variables (e.g., `user_input`, `question`).\n"
    "Do not reassign variable names that match tool functions (e.g., weather_info, web_search, guest_info_retriever)\n"
    "Always include Python code inside triple backticks followed by <end_code> at the end of the block.\n"
    "You do not ask the user for input. If no data is available, assume dummy or default values.\n"
    "To extract guest names, only use lines starting with 'Name:' and strip the result.\n"
    "Never include explanation, thoughts, or Markdown inside code blocks.\n"
    "Do not prefix code with Thought: or Code:. Only pure Python code is accepted inside\n"
)


class OllamaModel:
    def __init__(self, model_name="mistral:instruct", url="http://localhost:11434"):
        self.model_name = model_name
        self.url = url

    def generate(self, prompt, stop_sequences=None, **kwargs):
        # Gestion format chat vs prompt simple
        if isinstance(prompt, list):
            messages = []
            messages.append({"role": "system", "content": SYSTEM_PROMPT})

            for msg in prompt:
                role = msg.get("role", "user")
                content = ""
                if isinstance(msg.get("content"), list):
                    content = "".join(part.get("text", "") for part in msg["content"])
                elif isinstance(msg.get("content"), str):
                    content = msg["content"]
                messages.append({
                    "role": role,
                    "content": content
                })

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False
            }
            if stop_sequences:
                payload["stop"] = stop_sequences

            response = requests.post(f"{self.url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        else:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            if stop_sequences:
                payload["stop"] = stop_sequences

            response = requests.post(f"{self.url}/api/generate", json=payload)
            response.raise_for_status()
            content = response.json()["response"]

        # ✅ Objet de retour avec .content et .token_usage.input_tokens
        class TokenUsage:
            def __init__(self, prompt_tokens=0, completion_tokens=0):
                self.input_tokens = prompt_tokens
                self.output_tokens = completion_tokens
                self.total_tokens = prompt_tokens + completion_tokens

        class Result:
            def __init__(self, content):
                self.content = content
                # Valeur simulée : tu peux améliorer avec un vrai compteur
                self.token_usage = TokenUsage(prompt_tokens=len(content.split()), completion_tokens=0)

        return Result(content)
