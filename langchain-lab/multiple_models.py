import os
from langchain_openai import ChatOpenAI

base_url = os.environ.get("OPENAI_API_BASE")

# Different models for different purposes

# 1. Fast & Low-cost for simple tasks
fast_model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    base_url=base_url
)

# 2. Coding expert for programming
coding_model = ChatOpenAI(
    model="alibaba/qwen3-coder-plus",
    temperature=0,
    base_url=base_url
)

# 4. Conversational for chatting
chat_model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0.7,
    base_url=base_url
)

# Test each model with appropriate tasks
print("=== FAST MODEL (Simple Math) ===")
print(fast_model.invoke("What is 25 * 4?").content)

print("\n=== CODING MODEL (Write Code) ===")
code_task = "Write a Python function to reverse a string"
print(coding_model.invoke(code_task).content[:250] + "...")

print("\n=== CHAT MODEL (Friendly Talk) ===")
print(chat_model.invoke("Tell me a fun fact about dolphins!").content)

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/multiple-models.txt', 'w') as f:
    f.write("MULTIPLE_MODELS_COMPLETE")