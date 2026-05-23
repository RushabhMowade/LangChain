from langchain_huggingface import ChatHuggingFace,HuggingfacePipeline



llm = HuggingfacePipeline.from_model_id(model_id = "meta-llama/Llama-3.2-1B-Instruct",task="text-generation",pipeline_kwargs =dict(temperature=0.5,max_new_tokens=50)

)
model =ChatHuggingFace(llm=llm)
result = model.invoke("What is RLHF")
print(result)