#agent.py

import os
import requests
import re

class BasicAgent:
    def __init__(self, model="llama3:instruct", prompt_filename="system_prompt.txt"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

        # Construire le chemin absolu depuis le fichier agent.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(base_dir, prompt_filename)

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def _query_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()



    def __call__(self, question: str) -> str:
        full_prompt = self.system_prompt.strip() + "\n\nQuestion: " + question + "\nAnswer:"
        print("\n📨 Prompt envoyé à Ollama :\n" + full_prompt + "\n")

        response = self._query_ollama(full_prompt)

        print("\n📤 Réponse complète reçue :\n" + response + "\n")

        # Extraire le contenu entre FINAL ANSWER: et fin de ligne
        match = re.search(r"FINAL ANSWER:\s*(.*)", response, re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
            print(f"✅ Réponse extraite : {final_answer}")
            return final_answer
        else:
            print("⚠️ Aucune réponse finale trouvée. Réponse brute renvoyée.")
            return response.strip()


