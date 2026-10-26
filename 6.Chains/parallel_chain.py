from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

prompt1 = PromptTemplate(
    template="Generate short notes on {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate five multiple-choice questions from the following:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template=(
        "Based on these notes and quiz, recommend related topics to learn next.\n"
        "Notes:\n{topic}\n\n"
        "Quiz:\n{quiz}"
    ),
    input_variables=["topic", "quiz"]
)

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.5,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "topic": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

chain = parallel_chain | prompt3 | model | parser

text = """
Artificial Intelligence (AI) enables machines to perform tasks that normally require human intelligence.

Machine Learning (ML) helps computers learn patterns from data.

Generative AI creates content such as text, images, and code.

RAG retrieves relevant information from documents to help an LLM generate context-based answers.

LangChain is a framework for building applications powered by language models.
"""

result = chain.invoke({"text": text})
print(result)


chain.get_graph().print_ascii()