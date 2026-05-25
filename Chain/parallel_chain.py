from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="conversational",
    temperature=0.5,
)
model2 = ChatHuggingFace(llm=llm)


text=""" Transformer is a neural network architecture used for various machine learning tasks, especially in natural language processing and computer vision. It focuses on understanding relationships within data to process information more effectively.

Uses attention mechanisms to capture relationships between inputs
Processes entire sequences at once instead of step by step
Improves performance on tasks involving context and dependencies
Widely used across NLP, vision and other AI applications
_what_are_transformers_.webp
Need For Transformers Model
Transformer architecture uses attention to process an entire sentence at once instead of reading words sequentially. This helps overcome limitations of models like RNNs and LSTMs that process data step by step.

Traditional models like RNNs (Recurrent Neural Networks) suffer from the vanishing gradient problem which leads to long-term memory loss.
RNNs process text sequentially meaning they analyze words one at a time.
For example:

 In the sentence: "XYZ went to France in 2019 when there were no cases of COVID and there he met the president of that country" the word "that country" refers to "France". 

However RNNs may struggle to capture long-range dependencies effectively, especially in long sequences, which can make linking distant words more difficult.

While adding more memory cells in LSTMs (Long Short-Term Memory networks) helped address the vanishing gradient issue they still process words one by one. This sequential processing means LSTMs can't analyze an entire sentence at once.

Traditional Seq2Seq models compress the entire input into a single fixed-size context vector.
This creates an information bottleneck, especially for long sequences.
Important context may be lost during compression, reducing performance.
Transformers solve this by using attention to access all tokens directly without compression.
For example:

 The word "point" has different meanings in these two sentences:

"The needle has a sharp point." (Point = Tip)
"It is not polite to point at people." (Point = Gesture)
"""


prompt1 = PromptTemplate(
    template='Generate notes for {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate Intermediate Quiz of 5 questions for {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz in a single document\n notes -> {notes},quiz ->{quiz}',
    input_variables=['notes','quiz']
)
parser = StrOutputParser()
parallel = RunnableParallel({
  'notes':prompt1 | model2 | parser,
  'quiz': prompt2 | model2 | parser}
 )
merge_chain = prompt3 | model1 | parser
final = parallel | merge_chain 

result = final.invoke({'text':text})

print(result)
chain.get_graph().print_ascii()