from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = " who is Rohit sharma"
docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query_e = embeddings.embed_query(text)
docs_e = embeddings.embed_documents(docs)

scores=cosine_similarity([query_e],docs_e)[0]
print(text)
print(f"Document no : {1 + np.argmax(scores)} With similarity score {scores[np.argmax(scores)]}")
print(f"Document: {docs[np.argmax(scores)]}")