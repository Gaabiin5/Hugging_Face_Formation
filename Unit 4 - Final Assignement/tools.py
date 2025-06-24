# tools.py

import math
import operator
import wikipedia
from langchain_community.document_loaders import WikipediaLoader

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
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

# --- Exécuteur d'actions ---



class ToolExecutor:
    """
    Exécute dynamiquement une action de type :
    Action: tool_name["arg1", "arg2"]
    en appelant la fonction correspondante enregistrée via le décorateur @tool.
    """

    @staticmethod
    def execute(tool_call: str) -> str:
        """
        Parse et exécute une commande Action de type :
        Action: tool_name["arg1", "arg2"]

        Retourne une chaîne de type :
        Observation: [résultat ou message d'erreur]
        """
        from . import REGISTERED_TOOLS  # Import depuis le registre global (défini en haut de tools.py)

        match = re.match(r'Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\[(.*)\]', tool_call.strip())
        if not match:
            return "Observation: Invalid action format."

        tool_name, args_string = match.groups()

        try:
            # Nettoyer les arguments
            args = [arg.strip().strip('"').strip("'") for arg in args_string.split(',') if arg.strip()]
        except Exception as e:
            return f"Observation: Argument parsing error: {e}"

        tool_fn = REGISTERED_TOOLS.get(tool_name)
        if not tool_fn:
            return f"Observation: Unknown tool: {tool_name}"

        try:
            result = tool_fn(*args)
            return f"Observation: {result}"
        except Exception as e:
            return f"Observation: Execution error: {e}"
