#agent.py

import os
import requests
import re
from tools import ToolExecutor  # Your tool registry and executor


# --- BasicAgent ---

class BasicAgent:
    def __init__(self, model="mistral:instruct", prompt_filename="system_prompt.txt"):
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

        # print("\n📤 Réponse complète reçue :\n" + response + "\n")

        # Extraire le contenu entre FINAL ANSWER: et fin de ligne
        match = re.search(r"FINAL ANSWER:\s*(.*)", response, re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
            # print(f"✅ Réponse extraite : {final_answer}")
            return final_answer
        else:
            print("⚠️ Aucune réponse finale trouvée. Réponse brute renvoyée.")
            return response.strip()

# --- ToolAgent ---

class ToolAgent:
    def __init__(self, model="mistral:instruct", prompt_file="system_prompt.txt"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

        # Load system prompt from file (contains tool usage instructions)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, prompt_file)
        with open(path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def query_ollama(self, prompt: str) -> str:
        """
        Sends a prompt to Ollama's local API and returns the response text.
        """
        response = requests.post(
            self.api_url,
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def extract_action(self, text: str) -> str | None:
        """
        Looks for an Action[...] in the model's response.
        Returns the action string if found, else None.
        """
        match = re.search(r"Action:\s*[a-zA-Z_][a-zA-Z0-9_]*\[.*\]", text)
        return match.group(0) if match else None

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extracts the FINAL ANSWER from the model's response, if present.
        """
        match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def __call__(self, question: str) -> str:
        """
        Main agent loop:
        1. Build the prompt,
        2. Loop until FINAL ANSWER is found,
        3. Return the final answer string.
        """

        # Step 1 – detect math-related keywords
        math_keywords = [
            "plus", "minus", "times", "multiplied", "divided", "difference",
            "sum", "percent", "percentage", "total", "average", "increase", "decrease"
        ]
        if any(word in question.lower() for word in math_keywords):
            tool_hint = "\n\nReminder: Use math tools (`add`, `multiply`, `subtract`, `divide`) to solve this."
        else:
            tool_hint = ""

        # Step 2 – assemble initial prompt with optional hint
        full_prompt = f"{self.system_prompt}\n\nQuestion: {question}{tool_hint}"
        history = full_prompt

        # Step 3 – run reasoning loop
        for step in range(5):
            print(f"\n📨 Prompt sent to Ollama (step {step+1}):\n{history}\n")

            response = self.query_ollama(history)
            print(f"\n📤 Model response:\n{response}\n")

            final_answer = self.extract_final_answer(response)
            if final_answer:
                print(f"✅ FINAL ANSWER found: {final_answer}")
                return final_answer

            action_call = self.extract_action(response)
            if action_call:
                observation = ToolExecutor.execute(action_call)
                print(f"🔧 Executed: {action_call} → {observation}")
                history += f"\n{response}\n{observation}"
            else:
                print("⚠️ No Action or FINAL ANSWER found. Stopping.")
                return "No valid action or final answer produced."

        return "FINAL ANSWER not found after 5 steps."

