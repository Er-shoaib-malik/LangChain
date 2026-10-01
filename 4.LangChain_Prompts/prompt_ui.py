from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

st.header("Research tool")

paper_input = st.selectbox( "Select Research paper name" , ["Attention is All you Need", "BERT: Pre-training of deep Bidirectional Transformers" ,"GPT-3 : Language Models are Few -shot Learners", "Diffusion Models Beaet Gans on image synthesis"])
style_input = st.selectbox( "Select  Explanation Style", ["Beginner-Friendly" ,"Technincal","code priented","Mathematical"])
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2) paragraphs","Medium (3-5 paragraphs)","Long (Detailed Explanation)"])

template = load_prompt('../template.json')

# prompt = template.invoke({
#     'paper_input' : paper_input ,
#     'style_input' : style_input ,
#     'length_input' : length_input
# })

if st.button("Summarize") :
    try:
        chain = template | llm
        result = chain.invoke({
            'paper_input' : paper_input ,
            'style_input' : style_input ,
            'length_input' : length_input
        })
        st.write(result.content)

    except Exception as e:
        st.error("Gemini is temporarily unavailable. Please try again in a moment.")
        print(e)




