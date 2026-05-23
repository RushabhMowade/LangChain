from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = " Delhi is the capital of india"
#docs=["hello boy","boy hello"]
vector = embeddings.embed_query(text)
#vector = embeddings.embed_documents(docs)

print(str(vector))