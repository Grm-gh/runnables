from langchain.tools import tool

@tool
def get_greeting(name:str)->str:
    """generating a greeting massage for a user"""
    return f"Hello {name},welcome to the AI world"


result=get_greeting.invoke({"name":"Gyanu"})
print(result)
print(get_greeting.args)