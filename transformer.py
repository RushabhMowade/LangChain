from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="gpt2"
)

result = pipe("What is LLM?", max_new_tokens=50)

print(result[0]["generated_text"])