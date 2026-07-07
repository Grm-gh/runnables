from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print()
search_tool = TavilySearchResults(max_result=3)
model =ChatMistralAI(model="mistral-small-latest")
prompt=ChatPromptTemplate.from_template(
    """
    you are helpful assitant which
    summarizes the following news into bullet points
    {news}
    """
)

chain=prompt | model |StrOutputParser() 
news_result=search_tool.run("latest AI news of 2026")
result=chain.invoke(news_result)
print(result)