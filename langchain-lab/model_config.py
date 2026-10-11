import os
from langchain_openai import ChatOpenAI

base_url = os.environ.get("OPENAI_API_BASE")

# Precise model for facts (low temperature)
precise_model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,      # Very consistent answers
    max_tokens=150,    # Limit response length
    base_url=base_url
)

# Creative model for stories (high temperature)
creative_model = ChatOpenAI(
    model="google/gemini-2.5-flash",
    temperature=0.9,    # Very creative
    max_tokens=200,
    base_url=base_url
)

# Test both behaviors with same prompt
prompt = "Describe a rainbow"

print("=== PRECISE MODEL (temp=0) ===")
print(precise_model.invoke(prompt).content)

print("\n=== CREATIVE MODEL (temp=0.9) ===")
print(creative_model.invoke(prompt).content)

# Streaming for real-time responses
streaming_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0.5,
    streaming=True,  # Enable streaming
    base_url=base_url
)

print("\n=== STREAMING RESPONSE ===")
for chunk in streaming_model.stream("Write a haiku about coding"):
    print(chunk.content, end="", flush=True)
print()  # New line after streaming

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/config-complete.txt', 'w') as f:
    f.write("CONFIG_COMPLETE")