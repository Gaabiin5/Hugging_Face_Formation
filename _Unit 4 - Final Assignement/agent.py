#agent.py

import os
import requests
import re
from tools import ToolExecutor  # Your tool registry and executor
# from llm_utils import ollama_chat_completion as chat_completion
from llm_utils import safe_gemini_chat_completion as chat_completion


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
        self.used_tools = []
        self.chat_history = []

        # Load system prompt from file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, prompt_file)
        with open(path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def extract_action(self, text: str) -> str | None:
        match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\[(.*?)\]", text, re.DOTALL)
        return match.group(0) if match else None

    def extract_final_answer(self, text: str) -> str | None:
        matches = re.findall(r"FINAL ANSWER:\s*(.*)", text, re.IGNORECASE)
        for match in matches:
            answer = match.strip()
            if answer and not any(bad in answer.lower() for bad in [
                "sorry", "next question", "i don’t", "i cannot", "context", "okay", "understand"
            ]):
                return answer
        return None

    
    def _is_valid_answer(self, answer: str) -> bool:
        if not answer:
            return False
        answer_lower = answer.lower()
        vague_starts = [
            "i don't know", "unable", "not sure", "no information",
            "i cannot", "n/a", "insufficient"
        ]
        return not any(answer_lower.startswith(bad) for bad in vague_starts)    


    def query_model(self, message: str) -> str:
        messages = self.chat_history + [{"role": "user", "content": message}]
        response = chat_completion(system=self.system_prompt, messages=messages, model=self.model)

        # 🚫 Si le message contient un FINAL ANSWER clair, stoppons là
        if "FINAL ANSWER:" in response:
            # On ne l’ajoute pas à l’historique pour ne pas faire continuer le modèle
            return response.strip()
        
        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def __call__(self, question: str | dict, log: bool = False, max_steps: int = 10) -> str | dict:
        self.used_tools = []
        self.chat_history = []
        trace = ""

        if isinstance(question, dict):
            question_text = question["question"]
            file_name = question.get("file_name")
        else:
            question_text = question
            file_name = None

        file_context = ""
        if file_name:
            extension = os.path.splitext(file_name)[-1].lower()
            if extension in [".txt"]:
                file_context = ToolExecutor.execute(f'Action: read_txt["{file_name}"]')
                self.used_tools.append("read_txt")
            elif extension in [".pdf"]:
                file_context = ToolExecutor.execute(f'Action: read_pdf["{file_name}"]')
                self.used_tools.append("read_pdf")
            elif extension in [".png", ".jpg", ".jpeg"]:
                file_context = ToolExecutor.execute(f'Action: read_image_description["{file_name}"]')
                self.used_tools.append("read_jpeg")
            # Possibilité d'ajouter CSV, JSON, etc.
            if log:
                trace += f"\n📎 File context loaded for {file_name}:\n{file_context}\n"

        intro_message = f"Context:\n{file_context}\n\nQuestion: {question_text}"
        if log:
            trace += f"📨 Initial question to model:\n{intro_message}\n"

        final_answer = None  # <== ✅ pour break + retour en dehors de boucle

        for step in range(max_steps):
            model_output = self.query_model(intro_message if step == 0 else "Continue.")
            if log:
                trace += f"\n📤 Step {step+1} - Model output:\n{model_output}\n"

            action_call = self.extract_action(model_output)
            if log:
                trace += f"\n🔎 action_call (raw extract): {action_call}\n"
            if action_call:
                tool_name = action_call.split("[")[0].split(":")[-1].strip()
                self.used_tools.append(tool_name)
                observation = ToolExecutor.execute(action_call)
                if log:
                    trace += f"\n🔧 Tool executed: {action_call} → {observation}\n"
                intro_message = observation
                # continue

            final_answer = self.extract_final_answer(model_output)
            if log:
                trace += f"\n🔎 FINAL ANSWER (raw extract): {final_answer}\n"

            if final_answer:
                # Vérification stricte
                if self._is_valid_answer(final_answer):
                    if log:
                        trace += f"\n✅ FINAL ANSWER accepted. Stopping.\n"
                    break
                else:
                    if log:
                        trace += f"\n⚠️ FINAL ANSWER rejected as invalid (likely fallback).\n"
                    final_answer = None  # reset pour continuer



            # 🔁 Sinon, on relance avec un rappel
            intro_message = "Reminder: Use at least one tool, then give FINAL ANSWER."

        # === FIN DE BOUCLE ===
        if final_answer:
            if log:
                return {
                    "final_answer": final_answer,
                    "used_tools": self.used_tools,
                    "trace": trace.strip()
                }
            return final_answer

        if log:
            trace += "\n🛑 FINAL ANSWER not found after maximum steps.\n"
            return {
                "final_answer": "FINAL ANSWER not found after maximum steps.",
                "used_tools": self.used_tools,
                "trace": trace.strip()
            }
        return "FINAL ANSWER not found after maximum steps."




