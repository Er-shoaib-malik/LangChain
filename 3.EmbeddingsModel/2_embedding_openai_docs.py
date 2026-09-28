from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

document = [
    "DELhi is the capital of india",
    "kolkata is the capital of West bengal",
    "Paris uis the capital of France"
]

result = embedding.embed_documents(document)

print(str(result))