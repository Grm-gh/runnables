from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel,RunnableLambda

#components
model = ChatMistralAI(model="mistral-small-latest")
parser = StrOutputParser()

# two different prompt templates

short_prompt = ChatPromptTemplate.from_template("explain {topic} in 2 lines")
detailed_prompt= ChatPromptTemplate.from_template("explain {topic} in deatil")

#input
topic ="Ai engineering"

# runnable lambda
chains=RunnableParallel({
    "short":RunnableLambda(lambda x:x['short']) |short_prompt | model | parser,
    "detailed": RunnableLambda(lambda y:y['long']) | detailed_prompt | model | parser
})
result=chains.invoke({
    "short":{ "topic":"ai engineering"},
    "long" :{"topic:ML engineering"}
})
print(result['short'])
print(result['detailed'])
