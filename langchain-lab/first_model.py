import os
from langchain_openai import ChatOpenAI

# Environment variables OPENAI_API_KEY and OPENAI_API_BASE are pre-configured

# Initialize your first model - ultra fast!
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Test it with a simple question
response = model.invoke("Hello! What's 2+2?")
print("User: Hello! What's 2+2?")
print(f"AI: {response.content}")

# Try another question
response = model.invoke("What's the capital of France?")
print("\nUser: What's the capital of France?")
print(f"AI: {response.content}")

# Save progress
with open('/Users/lilywu/github/LongChain example 2/langchain-lab/first-model.txt', 'w') as f:
    f.write("FIRST_MODEL_COMPLETE")