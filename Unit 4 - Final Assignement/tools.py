# tools.py

import math
import operator
import wikipedia
from langchain_community.document_loaders import WikipediaLoader
from duckduckgo_search import DDGS
from llm_utils import ollama_chat_completion


import re

# --- Creation du decorateur 

from typing import Callable, Dict

REGISTERED_TOOLS: Dict[str, Callable] = {}

def tool(func: Callable) -> Callable:
    """Décorateur pour enregistrer une fonction comme outil disponible pour l'agent."""
    REGISTERED_TOOLS[func.__name__] = func
    return func


# --- Définition des outils disponibles ---

@tool
def add(a: str, b: str) -> int:
    return int(a) + int(b)

@tool
def multiply(a: str, b: str) -> int:
    return int(a) * int(b)

@tool
def subtract(a: str, b: str) -> int:
    return int(a) - int(b)

@tool
def divide(a: str, b: str) -> float:
    b = int(b)
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return int(a) / b

@tool
def modulus(a: str, b: str) -> int:
    return int(a) % int(b)

@tool
def wiki_search(query: str) -> str:
    """
    Searches Wikipedia and returns a concise summary (max 2 sentences).
    """
    try:
        # Optionally, set language if needed
        wikipedia.set_lang("en")

        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: Try a more specific term. Suggestions: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "No page found for the query."
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool
def web_search(query: str) -> str:
    """
    Searches the web via DuckDuckGo and returns the most relevant snippet.
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=1)
            first = results[0] if results else None
            if first and "body" in first:
                return first["body"]
            elif first and "snippet" in first:
                return first["snippet"]
            else:
                return "No relevant result found."
    except Exception as e:
        return f"Error in web_search: {e}"
    
@tool
def extract_answer(text: str, question: str) -> str:
    """
    Extract the best short answer to the question from the given context using the local LLM.
    """
    return ollama_chat_completion(
        system="You are a precise QA assistant. Answer with as few words as possible.",
        messages=[{"role": "user", "content": f"Context:\n{text}\n\nQuestion: {question}\nAnswer:"}]
    )



# --- Exécuteur d'actions ---



import re

class ToolExecutor:
    """
    Exécute dynamiquement une action de type :
    Action: tool_name["arg1", "arg2"]
    en appelant la fonction correspondante enregistrée via le décorateur @tool.
    """

    @staticmethod
    def is_unsatisfactory(text: str) -> bool:
        """
        Détermine si le résultat est insuffisant pour être exploitable.
        """
        if not text or len(text.strip()) < 10:
            return True
        text = text.lower()
        return any(bad in text for bad in ["no result", "not found", "unable", "could not", "empty", "none"])

    @staticmethod
    def execute(tool_call: str) -> str:
        """
        Parse et exécute une commande Action de type :
        Action: tool_name["arg1", "arg2"]

        Si l'outil principal échoue ou donne une réponse non satisfaisante, 
        bascule automatiquement sur un outil de fallback (ex. web_search).
        """

        match = re.match(r'Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\[(.*)\]', tool_call.strip())
        if not match:
            return "Observation: Invalid action format."

        tool_name, args_string = match.groups()

        try:
            args = [arg.strip().strip('"').strip("'") for arg in args_string.split(',') if arg.strip()]
        except Exception as e:
            return f"Observation: Argument parsing error: {e}"

        tool_fn = REGISTERED_TOOLS.get(tool_name)
        if not tool_fn:
            return f"Observation: Unknown tool: {tool_name}"

        try:
            result = tool_fn(*args)

            # ✅ Si le résultat est insatisfaisant, fallback sur web_search
            if tool_name == "wiki_search" and "web_search" in REGISTERED_TOOLS and ToolExecutor.is_unsatisfactory(str(result)):
                try:
                    fallback_fn = REGISTERED_TOOLS["web_search"]
                    fallback_result = fallback_fn(*args)
                    return f"Observation: {fallback_result}"
                except Exception as e:
                    return f"Observation: Fallback error: {e}"

            return f"Observation: {result}"

        except Exception as e:
            return f"Observation: Execution error: {e}"
