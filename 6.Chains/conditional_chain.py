from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser , JsonOutputParser
from langchain_core.runnables import RunnableParallel ,RunnableBranch
from pydantic import BaseModel ,Field
from typing import Literal


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.1,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description="Give the sentiment of the feedback")

parser2 = JsonOutputParser()

# prompt1 = PromptTemplate(
#     template="""Classify this feedback as positive or negative.
# Feedback: {feedback}
# Return only a JSON object, for example:
# {{"sentiment": "positive"}}
# or
# {{"sentiment": "negative"}}
# {format_instructions}""",
#     input_variables=["feedback"],
#     partial_variables={
#         "format_instructions": parser2.get_format_instructions()
#     }
# )

# chain = prompt1 | model | parser2

# result = chain.invoke({
#     "feedback": "This is a wonderful smartphone"
# })

# print(result.sentiment)



prompt1 = PromptTemplate(
    template="""Classify the sentiment of this feedback.
Feedback: {feedback}

Reply with exactly one JSON object.
Use "positive" for good feedback and "negative" for bad feedback.
Example response: {{"sentiment": "positive"}}

JSON response:""",
    input_variables=["feedback"]
)

chain = prompt1 | model | parser2

result = chain.invoke({
    "feedback": "This is a wonderful smartphone"
})

print("output:")
print(result["sentiment"])

chain.get_graph().print_ascii()