import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict

# Initialize model
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

parser = StrOutputParser()

# Example 1: Streaming responses
print("=== Streaming Responses ===")
print("Tokens will appear as they're generated:\n")

streaming_prompt = ChatPromptTemplate.from_template(
    "Tell me a story about {topic} in 3 sentences"
)

streaming_chain = streaming_prompt | model | parser

# Stream the response
for chunk in streaming_chain.stream({"topic": "a brave robot"}):
    print(chunk, end="", flush=True)
print("\n")

# Example 2: Batch processing
print("\n=== Batch Processing ===")

batch_prompt = ChatPromptTemplate.from_template(
    "Define {word} in one sentence"
)

batch_chain = batch_prompt | model | parser

# Process multiple inputs at once
words = [
    {"word": "LangChain"},
    {"word": "LCEL"},
    {"word": "Embeddings"},
    {"word": "Vector Store"}
]

print("Processing batch of words...")
batch_results = batch_chain.batch(words)

for word, result in zip(words, batch_results):
    print(f"\n{word['word']}: {result}")

# Example 3: Async execution
print("\n=== Async Execution ===")

async def run_async_chains():
    """Run multiple chains asynchronously"""

    async_prompt = ChatPromptTemplate.from_template(
        "Generate a {type} about {topic}"
    )

    async_chain = async_prompt | model | parser

    # Define multiple tasks
    tasks = [
        async_chain.ainvoke({"type": "haiku", "topic": "coding"}),
        async_chain.ainvoke({"type": "limerick", "topic": "debugging"}),
        async_chain.ainvoke({"type": "quote", "topic": "learning"})
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    types = ["Haiku", "Limerick", "Quote"]
    for type_name, result in zip(types, results):
        print(f"\n{type_name}:\n{result}")

# Run async example
print("Running async chains...")
asyncio.run(run_async_chains())

# Example 4: Error handling with fallbacks
print("\n=== Error Handling with Fallbacks ===")

def might_fail(input_dict: Dict) -> str:
    """Function that might fail for demonstration"""
    text = input_dict.get("text", "")
    if "error" in text.lower():
        raise ValueError("Simulated error!")
    return f"Successfully processed: {text}"

def fallback_handler(input_dict: Dict) -> str:
    """Fallback function when main fails"""
    return f"Fallback response for: {input_dict.get('text', 'unknown')}"

# Create chains with error handling
risky_chain = RunnableLambda(might_fail)
safe_chain = RunnableLambda(fallback_handler)

# Use with_fallbacks for error handling
robust_chain = risky_chain.with_fallbacks([safe_chain])

# Test with normal input
normal_result = robust_chain.invoke({"text": "normal input"})
print(f"Normal input result: {normal_result}")

# Test with error-triggering input
error_result = robust_chain.invoke({"text": "This has an error in it"})
print(f"Error input result: {error_result}")

# Example 5: Chain configuration with bind
print("\n=== Configuration with .bind() ===")

# Create a configurable chain
configurable_prompt = ChatPromptTemplate.from_template(
    "You are a {personality} assistant. Respond to: {query}"
)

# Bind different configurations
friendly_chain = (
    configurable_prompt.partial(personality="friendly and helpful")
    | model.bind(temperature=0.9)
    | parser
)

professional_chain = (
    configurable_prompt.partial(personality="professional and formal")
    | model.bind(temperature=0.3)
    | parser
)

query = {"query": "How can I improve my coding skills?"}

print("\nFriendly response:")
print(friendly_chain.invoke(query)[:150] + "...")

print("\nProfessional response:")
print(professional_chain.invoke(query)[:150] + "...")

# Example 6: Using .map() for element-wise operations
print("\n=== Using .map() for Lists ===")

def process_item(item: str) -> str:
    """Process individual item"""
    return f"Processed: {item.upper()}"

# Create a chain that processes lists
list_processor = RunnableLambda(process_item)

# Apply to each element in a list
items = ["apple", "banana", "cherry"]
processed = list_processor.map().invoke(items)

print(f"Original: {items}")
print(f"Processed: {processed}")

print("\n✅ All LCEL advanced features demonstrated!")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/advanced-lcel.txt', 'w') as f:
    f.write("ADVANCED_LCEL_COMPLETE")