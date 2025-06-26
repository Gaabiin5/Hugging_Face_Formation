from smolagents import DuckDuckGoSearchTool
from smolagents import Tool
import random
from huggingface_hub import list_models


# Initialize the DuckDuckGo search tool
search_tool = DuckDuckGoSearchTool()


import requests
from datetime import datetime
from smolagents.tools import Tool

class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches real weather forecast or historical weather for a given location using Open-Meteo API."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for."
        },
        "date": {
            "type": "string",
            "description": "The optional date to get weather info for, in 'YYYY-MM-DD' format.",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, location: str, date : str=None):
        # Get lat/lon for the location
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": location, "count": 1}
        geo_response = requests.get(geo_url, params=geo_params)
        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Location '{location}' not found."

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]


        # Format date
        if date:
            try:
                parsed_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                try:
                    parsed_date = datetime.strptime(date, "%B %d, %Y")
                except ValueError:
                    return "Invalid date format. Use 'YYYY-MM-DD' or 'June 23, 2025'."
        else:
            parsed_date = datetime.today()

        date_str = parsed_date.strftime("%Y-%m-%d")

        # Get forecast
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weathercode"],
            "timezone": "auto",
            "start_date": date_str,
            "end_date": date_str
        }

        weather_response = requests.get(weather_url, params=weather_params)
        weather_data = weather_response.json()

        try:
            temp_max = weather_data["daily"]["temperature_2m_max"][0]
            temp_min = weather_data["daily"]["temperature_2m_min"][0]
            precipitation = weather_data["daily"]["precipitation_sum"][0]
            summary = f"Weather in {location} on {date_str}: {temp_min}–{temp_max}°C, Precipitation: {precipitation}mm"
        except Exception as e:
            return f"Could not retrieve weather data for {location} on {date_str}."

        return summary



class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from."
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))
            
            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"

