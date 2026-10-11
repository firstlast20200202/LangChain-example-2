from langchain_core.prompts import PromptTemplate

# 1. Create a basic template
template = PromptTemplate(
    input_variables=["product", "feature"],
    template="Generate a marketing slogan for {product} highlighting {feature}."
)

# 2. Use the template
prompt = template.format(product="LangChain", feature="AI orchestration")
print("Generated prompt:", prompt)

# 3. Try different variables
examples = [
    {"product": "Smartphone", "feature": "camera quality"},
    {"product": "Electric Car", "feature": "eco-friendly"},
    {"product": "AI Assistant", "feature": "natural conversation"}
]

for example in examples:
    result = template.format(**example)
    print(f"• {result}")

# Save progress
with open('/Users/lilywu/github/LongChain example 2/langchain-lab/basic-templates.txt', 'w') as f:
    f.write("BASIC_TEMPLATES_COMPLETE")