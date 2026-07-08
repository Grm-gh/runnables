from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print

#1 creating a tool

@tool
def get_text_length(text:str)->int:
    """returns the number of character in given string"""
    return len(text)

model =ChatMistralAI(model="mistral-small-latest")

#tool binding
llm_with_tool=model.bind_tools([get_text_length])
result=llm_with_tool.invoke("Retun number of character in a given text : 'hello how are you'")

print(result.tool_calls[0])

result=get_text_length.invoke({'name': 'get_text_length', 'args': {'text': 'hello how are you'}, 'id': 'A3tKQRHV0', 'type': 'tool_call'})
print(result)


