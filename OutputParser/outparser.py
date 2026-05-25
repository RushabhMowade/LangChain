from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel,Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="conversational",
    temperature=0.5,
)


model = ChatHuggingFace(llm=llm)


"""

template1 = PromptTemplate(
    template='write the a report on {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='write the a 5 line summary on {text}',
    input_variables=['text']
)

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

"""


parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me some sunshine give me some prays \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
#prompt=template.format()
#result = model.invoke(prompt)
#final = parser.parse(result.content)
chain = template | model | parser
final = chain.invoke({})
print(final)


"""

"""


# Structured Output
schema = [
    ResponseSchema(name='Fact 1',description='Fact 1 about the topic'),
    ResponseSchema(name='Fact 2',description='Fact 2 about the topic'),
    ResponseSchema(name='Fact 2',description='Fact 3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about the {topic} \n {format_instruction}',
    input_variables=["topic"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({"topic":"Black Hole"})
print(result)

"""

class Person(BaseModel):
    name: str = Field(...,description="Name of the Person")
    age : int = Field(...,gt=18,description="Age of the person]")
    city : str = Field(...,description="City")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name ,age and city of {topic} Person \n {format_instruction}',
    input_variables=["topic"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'German'})
result = model.invoke(prompt)
final = parser.parse(result.content)
#chain = template | model | parser
#final = chain.invoke({'topic': 'German'})  
print(final)