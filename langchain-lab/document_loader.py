from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Create a sample knowledge base document
knowledge_doc = """
LangChain is a framework for developing applications powered by language models.

Key Features:
1. Chains: Connect multiple LLM calls and tools in sequence
2. Memory: Maintain conversation context across interactions
3. Agents: Enable LLMs to use tools and make decisions
4. RAG: Retrieve and use external knowledge for better answers

LCEL (LangChain Expression Language) provides a declarative way to compose chains.
It uses the pipe operator (|) to connect components in a readable manner.

Best Practices:
- Use appropriate chunk sizes for your documents (typically 500-1000 chars)
- Overlap chunks to maintain context (20% overlap is common)
- Choose the right embedding model for your use case
- Implement proper error handling in production
"""

# Save the document
with open('/Users/lilywu/github/LongChain example 2/langchain-lab/knowledge_base.txt', 'w') as f:
    f.write(knowledge_doc)

# Load the document using modern LangChain
loader = TextLoader('/Users/lilywu/github/LongChain example 2/langchain-lab/knowledge_base.txt')
documents = loader.load()

print("📄 Document loaded successfully!")
print(f"   Total characters: {len(documents[0].page_content)}")

# Split into chunks using modern text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # Size of each chunk
    chunk_overlap=50,     # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"\n✂️ Document split into {len(chunks)} chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"\nChunk {i} ({len(chunk.page_content)} chars):")
    print(f"   {chunk.page_content[:80]}...")

# Alternative: Create documents directly
print("\n📚 Creating documents from strings:")
text_docs = [
    "LangChain makes it easy to build LLM applications.",
    "RAG combines retrieval with generation for better answers.",
    "Vector databases store embeddings for similarity search."
]

# Convert strings to Document objects
docs_from_strings = [Document(page_content=text) for text in text_docs]
print(f"   Created {len(docs_from_strings)} documents from strings")

with open('/Users/lilywu/github/LongChain example 2/langchain-lab/doc-loading.txt', 'w') as f:
    f.write("DOC_LOADING_COMPLETE")