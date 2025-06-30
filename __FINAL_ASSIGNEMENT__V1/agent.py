import os
import re

from dotenv import load_dotenv

# LangChain Core
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# LangGraph Core
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

# LangChain LLM Providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Local imports
from tools import tools, vector_store


# --- Chargement des differentes cles apis contenues dans le fichier .env

load_dotenv()



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
            trace_lines = []
            tools_used = []

            for i, m in enumerate(all_messages):
                trace_lines.append("=" * 60)
                trace_lines.append(f"[{i+1}] TYPE: {m.type.upper()}")

                if isinstance(m, HumanMessage):
                    trace_lines.append(f"Human: {m.content}")

                elif isinstance(m, SystemMessage):
                    trace_lines.append(f"System Prompt:\n{m.content}")

                elif isinstance(m, ToolMessage):
                    tool_name = getattr(m, "name", "Unknown Tool")  # .name peut exister selon la version
                    trace_lines.append(f"Tool response from: {tool_name}")
                    trace_lines.append(f"Content:\n{m.content}")
                    tools_used.append(tool_name)


                elif hasattr(m, "tool_calls") and m.tool_calls:
                    trace_lines.append("AI (tool-calling):")
                    for call in m.tool_calls:
                        tool_name = call.get("name", "UNKNOWN")
                        call_id = call.get("id", "N/A")
                        args = call.get("args", {})

                        trace_lines.append(f"  ↪ Tool: {tool_name}")
                        trace_lines.append(f"     ID: {call_id}")
                        trace_lines.append("     Args:")
                        for k, v in args.items():
                            trace_lines.append(f"       - {k}: {v}")
                        tools_used.append(tool_name)

                else:
                    trace_lines.append(f"AI Response:\n{m.content}")

            trace_lines.append("=" * 60)
            trace = "\n".join(trace_lines)

            return {
                "final_answer": final_answer,
                "used_tools": bool(tools_used),
                "tools_used": list(set(tools_used)),
                "trace": trace,
            }

    


# Load the system_prompt

base_dir = os.path.dirname(__file__)
prompt_path = os.path.join(base_dir, "system_prompt.txt")

with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()


# System message
sys_msg = SystemMessage(content=system_prompt)


# --- Building the graph

def build_graph(provider: str = "groq"):
    """Build the LangGraph reasoning agent"""

    # 1. Select the LLM based on the provider
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

    # 2. Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # 3. Define LangGraph nodes

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here is a similar Q&A for context:\n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    # 4. Build the LangGraph
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
