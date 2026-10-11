import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import time

# Initialize model
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

parser = StrOutputParser()

# Create different prompts for parallel execution
joke_prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)

fact_prompt = ChatPromptTemplate.from_template(
    "Tell me an interesting fact about {topic}"
)

poem_prompt = ChatPromptTemplate.from_template(
    "Write a 2-line poem about {topic}"
)

# Create individual chains
joke_chain = joke_prompt | model | parser
fact_chain = fact_prompt | model | parser
poem_chain = poem_prompt | model | parser

# Method 1: RunnableParallel with dictionary
print("=== Parallel Execution with RunnableParallel ===")
start_time = time.time()

parallel_chain = RunnableParallel(
    joke=joke_chain,
    fact=fact_chain,
    poem=poem_chain
)

# Execute all three chains in parallel
results = parallel_chain.invoke({"topic": "programming"})

print(f"\n⏱️ Execution time: {time.time() - start_time:.2f} seconds")
print(f"\n😄 Joke: {results['joke']}")
print(f"\n📚 Fact: {results['fact']}")
print(f"\n📝 Poem: {results['poem']}")

# Method 2: Using RunnableParallel with different inputs
print("\n=== Parallel with Different Inputs ===")

analysis_chain = RunnableParallel(
    sentiment=ChatPromptTemplate.from_template(
        "What's the sentiment of: {text}"
    ) | model | parser,

    summary=ChatPromptTemplate.from_template(
        "Summarize in 10 words: {text}"
    ) | model | parser,

    keywords=ChatPromptTemplate.from_template(
        "Extract 3 keywords from: {text}"
    ) | model | parser
)

text = "LangChain makes building LLM applications easy and fun!"
analysis = analysis_chain.invoke({"text": text})

print(f"Text: {text}")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Summary: {analysis['summary']}")
print(f"Keywords: {analysis['keywords']}")

# Method 3: Combining parallel and sequential
print("\n=== Combined Parallel and Sequential ===")

# First, run parallel chains to gather info
gather_info = RunnableParallel(
    pros=ChatPromptTemplate.from_template(
        "List 2 pros of {topic}"
    ) | model | parser,

    cons=ChatPromptTemplate.from_template(
        "List 2 cons of {topic}"
    ) | model | parser
)

# Then use the results in a summary
summary_prompt = ChatPromptTemplate.from_template(
    "Based on these pros: {pros}\nAnd these cons: {cons}\n"
    "Write a balanced conclusion."
)

# Combine parallel and sequential
combined_chain = gather_info | summary_prompt | model | parser

conclusion = combined_chain.invoke({"topic": "working from home"})
print(f"Conclusion: {conclusion}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/parallel-chains.txt', 'w') as f:
    f.write("PARALLEL_CHAINS_COMPLETE")