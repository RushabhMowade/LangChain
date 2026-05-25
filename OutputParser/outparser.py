from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="conversational",
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='write the a report on {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='write the a 5 line summary on {text}',
    input_variables=['text']
)
"""

prompt1=template1.invoke({'topic':'Balck Hole'})
result = model.invoke(prompt1)
prompt2= template2.invoke({'text':result.content})
print(model.invoke(prompt2).content)


"""
"""

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'Black Hole'})
print(result)

"""

parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me some sunshine give me some prays \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
prompt=template.format()
result = model.invoke(prompt)
print(parser.parse(result.content))