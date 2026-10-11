step 1: full stack installation
$ mkdir -p langchain-lab && cd langchain-lab
$ python3 -m venv venv && source venv/bin/activate
$ pip install --upgrade pip uv

$ uv pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface        //Core LangChain package

$ uv pip install langchain-openai langchain-anthropic langchain-google-genai        //LLM Providers

$ uv pip install faiss-cpu sentence-transformers      //Vector store and embeddings

$ $ uv pip install python-dotenv gradio     //UI and utilities

step 2: set up prompt templates
create 4 files and execute each such as $ python basic_template.py
basic_template.py
chat_template.py
few_shot_template.py
advanced_templates.py

step 3:
connect to AI models
create an openai api key at https://platform.openai.com/api-keys
run $ export OPENAI_API_KEY="your key"
create 4 files and execute each such as $ python first_model.py
first_model.py
messages_demo.py
model_config.py
multiple_models.py
then you'll get this message:
"langchain_openai.chat_models.base.OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}"

step 4: chain 
create 4 files and execute each such as $ python sequential_chain.py
sequential_chain.py
parallel_chains.py
dynamic_routing.py
advanced_lcel.py

step 5: build contexts
create 2 files and execute each such as $ python memory_fundamentals.py
memory_fundamentals.py
advanced_memory.py

step 6: build RAG
create 2 files and execute each such as $ python document_loader.py
document_loader.py
retrieval_chain.py
