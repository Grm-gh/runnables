from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

# Model
model = ChatMistralAI(model="mistral-small-latest")

# Parser
parser = StrOutputParser()

# Store conversation history
history = []

print("🤖 Chatbot started! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Format prompt with previous history
    formatted_prompt = prompt.format_prompt(
        history=history,
        input=user_input
    )

    # Invoke model
    response = model.invoke(formatted_prompt)

    # Parse response
    ai_response = parser.parse(response)

    print("AI:", ai_response.content)

    # Save conversation to history
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=ai_response.content))