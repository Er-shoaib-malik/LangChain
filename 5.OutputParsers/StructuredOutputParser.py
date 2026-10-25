from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 500,
        "temperature": 0.5,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)

Schema = [
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(Schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic' :'black hole'})

result = model.invoke(prompt)

print("Raw output:", result.content)

try:
    final_result = parser.parse(result.content)
    print("Parsed output:", final_result)
except Exception as e:
    print("Parsing failed:", e)