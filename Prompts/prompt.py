import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import load_prompt


my_token = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = my_token

st.header("RESEARCH TOOL")

p = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

s = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

l = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
#####################################
import json
from langchain_core.load import loads

with open('Prompts/template.json', 'r', encoding='utf-8') as f:
    json_string = f.read()

template = loads(json_string)
#######################################



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    temperature=0.5,
    huggingfacehub_api_token=my_token
)


chat_model = ChatHuggingFace(llm=llm)

if st.button("Summarize"):
    chain= template | chat_model
    result = chain.invoke({
    'paper_input':p,
    'style_input':s,
    'length_input':l
})
    st.write(result.content)