from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-5.5')

result = llm.invoke("what is the capital of india")

print(result)