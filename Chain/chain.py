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


template1 = PromptTemplate(
    template='write the a report on {topic}',
    input_variables=['topic']
)
template2 = PromptTemplate(
    template='write the a 5 line summary on {text}',
    input_variables=['text']
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'Black Hole'})
print(result)
chain.get_graph().print_ascii()