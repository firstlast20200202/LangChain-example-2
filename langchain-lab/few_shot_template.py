from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define examples that teach the pattern
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
    {"input": "hot", "output": "cold"}
]

# Template for each example
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Few-shot template
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Find the opposite of each word:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

# Test the pattern learning
test_words = ["big", "light", "expensive", "difficult"]

print("🎯 Few-Shot Learning in Action:")
print("=" * 40)

for word in test_words:
    prompt = few_shot_template.format(word=word)
    print(f"\n📝 Generated Prompt for '{word}':")
    print(prompt)
    print("-" * 30)

# Advanced: Dynamic example selection
print("\n🔄 Advanced: Selective Examples")
selected_examples = examples[:2]  # Use only first 2 examples
dynamic_template = FewShotPromptTemplate(
    examples=selected_examples,
    example_prompt=example_template,
    prefix="Learn the pattern from these examples:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

prompt = dynamic_template.format(word="bright")
print(prompt)

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/few-shot-templates.txt', 'w') as f:
    f.write("FEW_SHOT_TEMPLATES_COMPLETE")