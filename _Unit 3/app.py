import gradio as gr
import random
from smolagents import GradioUI, CodeAgent, HfApiModel

from ollama_model import OllamaModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

# Initialize the Hugging Face model
# model = HfApiModel()
model = OllamaModel(model_name="llama3:instruct")  # ou "llama3", "mistral", etc.

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=2,   # Enable planning every 3 steps
    max_steps=10,
    additional_authorized_imports=["requests","json","re"]
)

if __name__ == "__main__":
    GradioUI(alfred).launch()