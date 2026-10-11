import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from typing import Dict, Any

# Initialize models with different capabilities
fast_model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

smart_model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

parser = StrOutputParser()

# Example 1: Simple transformation with RunnableLambda
print("=== Data Transformation with RunnableLambda ===")

def uppercase_text(text: str) -> str:
    '''Transform text to uppercase'''
    return text.upper()

def count_words(text: str) -> Dict[str, Any]:
    '''Count words and add metadata'''
    words = text.split()
    return {
        "original": text,
        "word_count": len(words),
        "processed": f"This text has {len(words)} words"
    }

# Chain with transformations
transform_chain = (
    RunnableLambda(uppercase_text)
    | RunnableLambda(count_words)
)

result = transform_chain.invoke("hello world from langchain")
print(f"Result: {result}")

# Example 2: Conditional routing based on input
print("\n=== Conditional Routing ===")

def route_by_length(input_dict: Dict) -> Any:
    '''Route to different chains based on input length'''
    text = input_dict.get("text", "")

    if len(text) < 20:
        # Short text: simple response
        prompt = ChatPromptTemplate.from_template(
            "Give a one-line response to: {text}"
        )
        chain = prompt | fast_model | parser
    else:
        # Long text: detailed analysis
        prompt = ChatPromptTemplate.from_template(
            "Provide a detailed analysis of: {text}"
        )
        chain = prompt | smart_model | parser

    return chain.invoke(input_dict)

routing_chain = RunnableLambda(route_by_length)

# Test with short input
short_result = routing_chain.invoke({"text": "Hi there!"})
print(f"\nShort input result: {short_result}")

# Test with long input
long_result = routing_chain.invoke({
    "text": "Explain the benefits of using LangChain for building LLM applications"
})
print(f"\nLong input result: {long_result[:200]}...")

# Example 3: Complex routing with multiple conditions
print("\n=== Multi-Condition Routing ===")

def classify_query(input_dict: Dict) -> str:
    '''Classify the type of query'''
    query = input_dict.get("query", "").lower()

    if any(word in query for word in ["code", "python", "function", "programming"]):
        return "technical"
    elif any(word in query for word in ["explain", "what", "why", "how"]):
        return "educational"
    elif any(word in query for word in ["joke", "funny", "humor"]):
        return "entertainment"
    else:
        return "general"

def route_by_type(input_dict: Dict) -> str:
    '''Route to specialized chains based on query type'''
    query_type = classify_query(input_dict)

    prompts = {
        "technical": "Provide a technical answer with code example for: {query}",
        "educational": "Explain in simple terms: {query}",
        "entertainment": "Respond humorously to: {query}",
        "general": "Answer: {query}"
    }

    prompt = ChatPromptTemplate.from_template(prompts[query_type])
    chain = prompt | smart_model | parser

    result = chain.invoke(input_dict)
    return f"[{query_type.upper()}] {result}"

smart_router = RunnableLambda(route_by_type)

# Test different query types
queries = [
    "How do I write a Python function?",
    "What is machine learning?",
    "Tell me something funny",
    "What's the weather like?"
]

for query in queries:
    result = smart_router.invoke({"query": query})
    print(f"\nQuery: {query}")
    print(f"Response: {result[:150]}...")

# Example 4: Chaining multiple lambdas
print("\n=== Chaining Multiple Lambdas ===")

def extract_topic(input_dict: Dict) -> Dict:
    '''Extract the main topic from input'''
    text = input_dict.get("text", "")
    # Simple topic extraction (in real app, might use NLP)
    words = text.split()
    topic = words[-1] if words else "unknown"
    return {"text": text, "topic": topic}

def enhance_with_context(input_dict: Dict) -> Dict:
    '''Add context based on topic'''
    topic = input_dict.get("topic", "")
    contexts = {
        "langchain": "In the context of LLM frameworks",
        "python": "In Python programming",
        "ai": "In artificial intelligence"
    }
    context = contexts.get(topic.lower(), "In general")
    return {**input_dict, "context": context}

# Build a chain with multiple lambda transformations
lambda_chain = (
    RunnableLambda(extract_topic)
    | RunnableLambda(enhance_with_context)
    | RunnableLambda(lambda x: f"{x['context']}, {x['text']}")
)

enhanced = lambda_chain.invoke({"text": "Tell me about langchain"})
print(f"Enhanced output: {enhanced}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/dynamic-routing.txt', 'w') as f:
    f.write("DYNAMIC_ROUTING_COMPLETE")