from langchain_core.prompts import ChatPromptTemplate

# Create a chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert with {years} years of experience."),
    ("human", "Explain {concept} to me in simple terms."),
    ("assistant", "I'll explain {concept} step by step.")
])

# Generate messages
messages = chat_template.format_messages(
    role="Python programming",
    years="10",
    concept="decorators"
)

print("🎭 Generated Chat Conversation:")
for msg in messages:
    role_color = {"system": "🔵", "human": "🟢", "assistant": "🟣"}
    icon = role_color.get(msg.type, "⚪")
    print(f"{icon} {msg.type.upper()}: {msg.content}")

# Try different scenarios
scenarios = [
    {"role": "Data Science", "years": "5", "concept": "machine learning"},
    {"role": "Web Development", "years": "8", "concept": "REST APIs"},
]

for scenario in scenarios:
    print(f"\n📋 Scenario: {scenario['concept']}")
    messages = chat_template.format_messages(**scenario)
    print(f"   System: {messages[0].content}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/chat-templates.txt', 'w') as f:
    f.write("CHAT_TEMPLATES_COMPLETE")