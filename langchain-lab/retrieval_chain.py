"""
Modern RAG Implementation using Pure LCEL
==========================================
This demonstrates the modern LCEL approach to building RAG systems.
"""

import os
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize model with proxy
model = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE"),
)

# Load HuggingFace embeddings (free, no API key needed!)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Create sample documents
print("📂 Creating vector store from documents...")
sample_docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_intro.txt"}
    ),
    Document(
        page_content="LCEL provides a declarative way to compose chains using the pipe operator.",
        metadata={"source": "lcel_guide.txt"}
    ),
    Document(
        page_content="Memory systems in LangChain help maintain conversation context.",
        metadata={"source": "memory_guide.txt"}
    ),
    Document(
        page_content="RAG combines retrieval with generation for accurate, grounded responses.",
        metadata={"source": "rag_guide.txt"}
    ),
    Document(
        page_content="Best practice: Use chunk sizes of 500-1000 characters for optimal retrieval.",
        metadata={"source": "best_practices.txt"}
    ),
]
print(f"📄 Created {len(sample_docs)} sample documents")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(sample_docs)

# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector store created!")

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """Use the following context to answer the question.
If you don't know the answer based on the context, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

# Helper to format documents
def format_docs(docs):
    """Format retrieved documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the LCEL RAG chain
# This chain: retrieves docs → formats them → generates answer
rag_chain = (
    RunnableParallel({
        "context": retriever,  # Retrieve relevant docs
        "question": RunnablePassthrough()  # Pass question through
    })
    .assign(formatted_context=lambda x: format_docs(x["context"]))  # Format docs
    .assign(
        answer=(
            {
                "context": itemgetter("formatted_context"),
                "question": itemgetter("question")
            }
            | prompt
            | model
            | StrOutputParser()
        )
    )
    .pick(["answer", "context"])  # Return answer and sources
)

# Test the RAG system
test_questions = [
    "What is LCEL?",
    "What are the key features of LangChain?",
    "What is the recommended chunk size for documents?",
    "How do I use memory in LangChain?"
]

print("\n🤖 RAG System Test")
print("=" * 60)

for question in test_questions:
    print(f"\n❓ Question: {question}")

    result = rag_chain.invoke(question)

    print(f"💡 Answer: {result['answer']}")

    # Show sources
    if result['context']:
        print(f"📚 Sources ({len(result['context'])} chunks used):")
        for i, doc in enumerate(result['context'][:2], 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:60]
            print(f"   {i}. [{source}] {preview}...")

print("\n✨ Modern RAG chain complete!")

# Write completion marker
with open("/Users/lilywu/github/LongChain example 2/langchain-lab/retrieval-chain.txt", "w") as f:
    f.write("RETRIEVAL_CHAIN_COMPLETE")
