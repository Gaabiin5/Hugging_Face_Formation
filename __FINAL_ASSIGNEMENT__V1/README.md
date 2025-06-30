---
title: GAIA Agent – Final Assignment
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: true
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# 🧠 GAIA Agent – Final Assignment

This project implements an autonomous agent designed to solve reasoning tasks from the [GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard), as part of the Hugging Face Agents course.

The agent uses a reasoning graph (LangGraph) built around a large language model (LLM) and enhanced with external tools (retrieval, search, math, etc.). The goal is to reach at least **30% accuracy** on GAIA tasks.

---

## 🚀 Project Structure

```
.
├── app.py                # Gradio interface for evaluation and submission
├── agent.py              # LangGraph + agent logic
├── tools.py              # All external tools and retriever setup
├── system_prompt.txt     # Prompt template with examples
├── supabase_docs.csv     # Optional: data to populate vector DB
├── test.ipynb            # Notebook to run and debug agent locally
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

---

## 🧠 Agent Capabilities

- **LLM Provider Options**: Google Gemini, Groq, Hugging Face
- **Step-by-step reasoning** via LangGraph
- **Tool usage**:
  - Math operations
  - Wikipedia and Arxiv search
  - Web search (Tavily)
  - Supabase retriever (semantic similarity search)

---

## 🧰 Setup

1. **Clone the repo** and create a virtual environment:

```bash
git clone <your-repo-url>
cd <your-project>
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Configure environment variables** (create a `.env` file):

```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
```

4. **Populate Supabase (if needed)**  
Follow the instructions here:  
👉 https://python.langchain.com/docs/integrations/vectorstores/supabase/

---

## 📊 Run Locally

Use the notebook or Python CLI to test:

```python
from agent import BasicAgent
agent = BasicAgent(provider="groq")
result = agent("What is 2 + 2", log=True)
print(result["final_answer"])
print(result["trace"])
```

---

## 🌐 Launch the Gradio App

```bash
python app.py
```

This allows you to log in with Hugging Face and evaluate the agent on GAIA tasks directly.

---

## 📈 Evaluation on GAIA

The agent will:

- Fetch the GAIA dataset via API
- Run reasoning + tools
- Submit results automatically
- Return score + logs

You can modify the agent behavior in `agent.py` and the tools in `tools.py`.

---

## 📎 Resources

- [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard)
- [LangGraph](https://www.langgraph.dev/)
- [LangChain](https://docs.langchain.com/)
- [Supabase](https://supabase.com/)

---

## 🧑‍💻 Author

Made with ❤️ as part of the Hugging Face Agents course.

Inspired by [this project](https://huggingface.co/spaces/baixianger/RobotPai/tree/main)