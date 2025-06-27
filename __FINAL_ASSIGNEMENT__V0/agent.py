#agent.py

"""LangGraph Agent"""
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
import re

load_dotenv()


from tools import tools, vector_store



# --- BasicAgent ---

class BasicAgent:
    """A langgraph agent."""
    def __init__(self,provider: str = "groq"):
        print("BasicAgent initialized.")
        self.graph = build_graph(provider=provider)

    def extract_final_answer(self, text):
        match = re.search(r"final answer\s*[:\-]?\s*(.*)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def __call__(self, question: str, log=False):
        messages = [HumanMessage(content=question)]
        result = self.graph.invoke({"messages": messages})

        all_messages = result["messages"]
        raw_answer = all_messages[-1].content
        final_answer = self.extract_final_answer(raw_answer)

        if log:
            trace = "\n".join([f"{m.type.upper()}: {m.content}" for m in all_messages])
            tools_used = [m.tool for m in all_messages if hasattr(m, "tool")]
            return {
                "final_answer": final_answer,
                "used_tools": bool(tools_used),
                "tools_used": tools_used,
                "trace": trace,
            }

        return final_answer
    


#Load the system_prompt

base_dir = os.path.dirname(__file__)
prompt_path = os.path.join(base_dir, "system_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()


# System message
sys_msg = SystemMessage(content=system_prompt)


# --- Construction du graphe 

def build_graph(provider: str = "groq"):
    """Build the LangGraph reasoning agent"""

    # 1. Sélection du LLM en fonction du provider
    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            )
        )
    else:
        raise ValueError("Invalid provider. Choose 'google', 'groq' or 'huggingface'.")

    # 2. Association des outils au LLM
    llm_with_tools = llm.bind_tools(tools)

    # 3. Définition des noeuds LangGraph

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here is a similar Q&A for context:\n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    # 4. Construction du graphe LangGraph
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
