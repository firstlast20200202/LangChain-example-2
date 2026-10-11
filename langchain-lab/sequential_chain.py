import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize model with proxy
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer this question: {question}"
)

# Create output parser
parser = StrOutputParser()

# Build the chain using LCEL pipe operator
chain = prompt | model | parser

# Execute the chain
result = chain.invoke({"question": "What is LCEL in LangChain?"})

print("User: What is LCEL in LangChain?")
print(f"AI: {result}")

# Chain with multiple steps
print("\n=== Multi-Step Chain ===")

# Step 1: Generate a topic
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a creative topic about {subject}"
)

# Step 2: Write a story about that topic  
story_prompt = ChatPromptTemplate.from_template(
    "Write a 2-sentence story about: {topic}"
)

# Build a multi-step chain
multi_chain = (
    {"topic": topic_prompt | model | parser}
    | story_prompt
    | model
    | parser
)

story = multi_chain.invoke({"subject": "robots"})
print(f"Generated Story: {story}")

# Using RunnablePassthrough to preserve input
print("\n=== Chain with Passthrough ===")

from langchain_core.runnables import RunnableParallel

chain_with_context = RunnableParallel(
    {
        "original": RunnablePassthrough(),
        "response": prompt | model | parser
    }
)

result_with_context = chain_with_context.invoke({"question": "What is 2+2?"})
print(f"Original Input: {result_with_context['original']}")
print(f"Response: {result_with_context['response']}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/sequential-chains.txt', 'w') as f:
    f.write("SEQUENTIAL_CHAINS_COMPLETE")