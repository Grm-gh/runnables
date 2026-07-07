from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

model =ChatMistralAI(model="mistral-small-latest")
parser=StrOutputParser()
code_prompt=ChatPromptTemplate.from_messages([
    ("system","you are a code generator"),
    ("human","{topic}")
])
explain_prompt=ChatPromptTemplate.from_messages([
    ("system","you are helpful assitant who explains code in simple text"),
    ("human","explain the code in words:\n{code}")
])

seq = code_prompt | model | parser 

seq2= RunnableParallel(
    {"code": RunnablePassthrough(),
     "explantion": explain_prompt | model | parser
     }
)
chain =seq| seq2
result= chain.invoke({"topic":"please write a code of palindrome in python"})
print(result['code'])
print(result['explantion'])