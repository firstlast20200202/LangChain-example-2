from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Template with Validation
def validate_topic_length(topic):
    if len(topic) > 50:
        raise ValueError("Topic too long!")
    return True

safe_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a summary about {topic}",
    validate_template=True
)

# 2. Partial Templates (Pre-filled variables)
partial_template = PromptTemplate(
    input_variables=["task"],
    template="As an expert {role}, please help with: {task}",
    partial_variables={"role": "AI consultant"}
)

# 3. Structured Output with Pydantic
class ProductReview(BaseModel):
    rating: int = Field(description="Rating from 1-5")
    pros: list[str] = Field(description="List of advantages")
    cons: list[str] = Field(description="List of disadvantages")
    recommendation: str = Field(description="Final recommendation")

parser = PydanticOutputParser(pydantic_object=ProductReview)

structured_template = PromptTemplate(
    template="Review this product: {product}\n{format_instructions}",
    input_variables=["product"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Conditional Templates
def create_conditional_template(user_level):
    if user_level == "beginner":
        template = "Explain {concept} in very simple terms with examples."
    elif user_level == "intermediate":
        template = "Explain {concept} with some technical details."
    else:  # advanced
        template = "Provide a comprehensive technical explanation of {concept}."

    return PromptTemplate(
        input_variables=["concept"],
        template=template
    )

# Test all features
print("⚡ Advanced Template Features Demo")
print("=" * 45)

# Test partial template
print("\n1️⃣ Partial Template:")
prompt1 = partial_template.format(task="optimize database queries")
print(f"   {prompt1}")

# Test structured output
print("\n2️⃣ Structured Output Template:")
prompt2 = structured_template.format(product="iPhone 15")
print(f"   {prompt2[:100]}...")

# Test conditional templates
print("\n3️⃣ Conditional Templates:")
for level in ["beginner", "intermediate", "advanced"]:
    template = create_conditional_template(level)
    prompt = template.format(concept="machine learning")
    print(f"   {level.upper()}: {prompt}")

# Best Practices Summary
print("\n🎯 Template Best Practices:")
best_practices = [
    "Always validate user inputs",
    "Use partial variables for constants",
    "Structure outputs with Pydantic",
    "Make templates conditional when needed",
    "Keep templates readable and maintainable"
]

for i, practice in enumerate(best_practices, 1):
    print(f"   {i}. {practice}")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/advanced-templates.txt', 'w') as f:
    f.write("ADVANCED_TEMPLATES_COMPLETE")