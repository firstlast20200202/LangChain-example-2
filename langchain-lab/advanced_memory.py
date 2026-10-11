import os
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage

# Initialize model (if not already initialized)
if 'model' not in locals():
    model = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )

# === 1. Summarization-Style Memory ===
print("=== Summary-Style Memory ===")

# Create prompt that asks model to summarize
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that summarizes conversations."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "First, briefly summarize the conversation so far, then respond.")
])

summary_chain = summary_prompt | model
summary_history = InMemoryChatMessageHistory()

summary_with_history = RunnableWithMessageHistory(
    summary_chain,
    lambda sid: summary_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Long conversation
topics = [
    "I'm a software engineer working on AI projects",
    "I use Python, JavaScript, and Rust",
    "My current project involves building a RAG system"
]

config = {"configurable": {"session_id": "summary-session"}}
for topic in topics:
    response = summary_with_history.invoke({"input": topic}, config)
    print(f"User: {topic}")
    print(f"AI: {response.content}\n")

# === 2. Window Memory - Keep Last K Messages ===
print("\n=== Window Memory (Last K Messages) ===")

# Window memory with manual trimming
window_history = InMemoryChatMessageHistory()
window_k = 2  # Keep last 2 exchanges

window_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

window_chain = window_prompt | model

# Custom wrapper to manage window size
def get_window_history(session_id):
    # Trim history to last k exchanges after adding
    if len(window_history.messages) > window_k * 2:
        window_history.messages = window_history.messages[-(window_k * 2):]
    return window_history

window_with_history = RunnableWithMessageHistory(
    window_chain,
    get_window_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Test with multiple messages
messages = ["My name is Bob", "I live in Seattle", 
            "I work at tech", "What's my name?"]

window_config = {"configurable": {"session_id": "window-session"}}
for msg in messages:
    response = window_with_history.invoke({"input": msg}, window_config)
    print(f"User: {msg}")
    print(f"AI: {response.content}")
    print(f"Memory size: {len(window_history.messages)} messages\n")

print("\n📈 Window Memory Summary:")
print(f"Window size: {window_k} exchanges")
print(f"Current messages in history: {len(window_history.messages)}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/advanced-memory.txt', 'w') as f:
    f.write("ADVANCED_MEMORY_COMPLETE")