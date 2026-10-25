
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.2,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate one fictional Indian person's name, age, and city.\n"
        "Return the actual person's data as JSON, not the schema.\n"
        "{format_instructions}\n"
        "JSON:"
    ),
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt = template.invoke({})
result = model.invoke(prompt)

print("Raw output:", result.content)

'''chain = template | model | parser
    chain.invoke({})
'''

try:
    final_result = parser.parse(result.content)
    print("Parsed output:", final_result)
    print("Name:", final_result.name)
    print("Age:", final_result.age)
    print("City:", final_result.city)
except Exception as e:
    print("Parsing failed:", e)
