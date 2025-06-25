#agent.py

import os
import requests
import re
from tools import ToolExecutor  # Your tool registry and executor


# --- BasicAgent ---

class BasicAgent:
    def __init__(self):
       print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

# --- ToolAgent ---

class ToolAgent:
    def __init__(self, model="mistral:instruct", prompt_file="system_prompt.txt"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.used_tools = []

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
        match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\[(.*?)\]", text, re.DOTALL)
        return match.group(0) if match else None

    def extract_final_answer(self, text: str) -> str | None:
        """
        Extracts the FINAL ANSWER from the model's response, if present.
        """
        match = re.search(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def __call__(self, question: str, log: bool = False, max_steps : int = 10) -> str | dict:
        """
        Main agent loop with optional logging.

        If log=True, returns a dict with:
        - final_answer
        - used_tools
        - trace
        """
        self.used_tools = []  # 🧹 Clear used tools for each call

        math_keywords = [
            "plus", "minus", "times", "multiplied", "divided", "difference",
            "sum", "percent", "percentage", "total", "average", "increase", "decrease"
        ]
        tool_hint = "\n\nReminder: Use math tools (`add`, `multiply`, `subtract`, `divide`) to solve this." \
            if any(word in question.lower() for word in math_keywords) else ""

        full_prompt = f"{self.system_prompt}\n\nQuestion: {question}{tool_hint}"
        history = full_prompt

        if log:
            trace = f"📨 Prompt sent to Ollama (step 0):\n{history}\n"

        self.used_tools = []  # reset for this question

        for step in range(max_steps):
            response = self.query_ollama(history)
            if log:
                trace += f"\n📤 Model response (step {step+1}):\n{response}\n"

            # 🚨 Anti-cheat: forbid final answer if no Action has been executed
            if "FINAL ANSWER:" in response and "Action:" not in history:
                history += f"\n{response}\nReminder: You must use at least one tool to answer. Try again."
                continue

            # 🛑 Garde-fou pour réponses vides ou floues
            if not response.strip() or "i don't know" in response.lower():
                history += "\nReminder: You must use one of the tools and give a final answer.\n"
                continue



            action_call = self.extract_action(response)
            if action_call:
                tool_name = action_call.split("[")[0].split(":")[-1].strip()
                self.used_tools.append(tool_name)
                observation = ToolExecutor.execute(action_call)
                history += f"\n{response}\n{observation}"
                if log:
                    trace += f"\n🔧 Executed: {action_call} → {observation}\n"
            else:
                history += f"\n{response}\nReminder: Use one of the tools (e.g. add, subtract, wiki_search). Try again."

            final_answer = self.extract_final_answer(response)
            if final_answer:
                if log:
                    return {
                        "final_answer": final_answer,
                        "used_tools": self.used_tools,
                        "trace": trace.strip()
                    }
                return final_answer


        if log:
            return {
                "final_answer": f"FINAL ANSWER not found after {max_steps} steps.",
                "used_tools": self.used_tools,
                "trace": trace.strip()
            }
        return f"FINAL ANSWER not found after {max_steps} steps."



