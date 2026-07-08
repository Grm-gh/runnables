from dotenv import load_dotenv
load_dotenv()

import os
import requests
from rich import print

from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

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

tools = {
    "get_weather": get_weather,
    "get_news": get_news,
}

llm_with_tools = llm.bind_tools(list(tools.values()))

# -----------------------------
# Agent Loop
# -----------------------------
messages = []

print("[bold green]City Intelligence System[/bold green]")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:

        result = llm_with_tools.invoke(messages)
        messages.append(result)

        # No tool needed
        if not result.tool_calls:
            print("\n[bold cyan]Assistant:[/bold cyan]")
            print(result.content)
            print()
            break

        # Execute tool(s)
        for tool_call in result.tool_calls:

            tool_name = tool_call["name"]

            print(f"\n[bold yellow]Tool Requested:[/bold yellow] {tool_name}")
            print(f"Arguments: {tool_call['args']}")

            confirm = input("Allow tool execution? (yes/no): ").strip().lower()

            if confirm not in ("yes", "y"):

                messages.append(
                    ToolMessage(
                        content="User denied permission to execute this tool.",
                        tool_call_id=tool_call["id"],
                    )
                )

                continue

            if tool_name not in tools:

                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' not found.",
                        tool_call_id=tool_call["id"],
                    )
                )

                continue

            # Execute tool
            tool_result = tools[tool_name].invoke(tool_call["args"])

            print("\n[green]Tool Output:[/green]")
            print(tool_result)

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"],
                )
            )