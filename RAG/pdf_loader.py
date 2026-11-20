from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('RAG/LAB 1 ASSIGNMENT.pdf')
docs = loader.load()

print(docs[0])