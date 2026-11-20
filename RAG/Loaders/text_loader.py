from langchain_community.document_loaders import TextLoader

loader = TextLoader('RAG/cricket.txt',encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0])