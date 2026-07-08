from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent

# -----------------------------
# API Keys
# -----------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# -----------------------------
# Weather Tool
# -----------------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"The current weather in {city} is {desc} with a temperature of {temp}°C."


# -----------------------------
# News Tool
# -----------------------------
@tool
def get_news(query: str) -> str:
    """Get the latest news about a city or topic."""

    response = tavily.search(
        query=f"latest news about {query}",
        topic="news",
        max_results=3,
    )

    if not response["results"]:
        return "No news found."

    news = []

    for item in response["results"]:
        news.append(
            f"Title: {item['title']}\n"
            f"URL: {item['url']}"
        )

    return "\n\n".join(news)


# -----------------------------
# LLM
# -----------------------------
llm = ChatMistralAI(model="mistral-small-latest")

agent=create_agent(
    llm,
    tools=[get_weather,get_news],
    system_prompt="you are a helpful city assistant"
)
print("city agent")
while True:
    user_input=input("you :")
    if user_input.lower=="exit":
        break
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result['messages'][-1].content)

