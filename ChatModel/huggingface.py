import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


my_token = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = my_token


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    temperature=0.5,
    huggingfacehub_api_token=my_token
)


chat_model = ChatHuggingFace(llm=llm)


print("Sending request to Hugging Face..")
try:
    result = chat_model.invoke("What is LLM?")
    print(result.content)
except Exception as e:
    print(f"\nError: {e}")