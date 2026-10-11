import os
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize model with proxy
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.7,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

# === 1. Simple In-Memory Chat History ===
print("=== Basic Chat Memory ===")

# Create in-memory chat history
chat_history = InMemoryChatMessageHistory()

# Define prompt with message history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
chain = prompt | model

# Wrap with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Have a conversation
config = {"configurable": {"session_id": "test-session"}}

response1 = chain_with_history.invoke(
    {"input": "My name is Alice and I love Python"},
    config
)
print(f"AI: {response1.content}")

response2 = chain_with_history.invoke(
    {"input": "What's my name and what do I love?"},
    config
)
print(f"AI: {response2.content}")

# Inspect memory contents
print("\n📝 Memory Contents:")
for msg in chat_history.messages:
    print(f"{msg.__class__.__name__}: {msg.content}")

# Save for verification
with open('/Users/lilywu/github/LongChain example 2/langchain-lab/memory-basics.txt', 'w') as f:
    f.write("MEMORY_BASICS_COMPLETE")