# step 1: full stack installation <br>
$ mkdir -p langchain-lab && cd langchain-lab <br>
$ python3 -m venv venv && source venv/bin/activate <br>
$ pip install --upgrade pip uv <br>

$ uv pip install langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface        //Core LangChain package <br>

$ uv pip install langchain-openai langchain-anthropic langchain-google-genai        //LLM Providers <br>

$ uv pip install faiss-cpu sentence-transformers      //Vector store and embeddings <br>

$ $ uv pip install python-dotenv gradio     //UI and utilities <br>

# step 2: set up prompt templates <br>
### create 4 files and execute each such as $ python basic_template.py <br>
basic_template.py <br>
chat_template.py <br>
few_shot_template.py <br>
advanced_templates.py <br>

# step 3: connect to AI models <br>
### create an openai api key at https://platform.openai.com/api-keys <br>
### run $ export OPENAI_API_KEY="your key" <br>
### create 4 files and execute each such as $ python first_model.py <br>
first_model.py <br>
messages_demo.py <br>
model_config.py <br>
multiple_models.py <br>
### then you'll get this message: <br>
"langchain_openai.chat_models.base.OpenAIRateLimitError: Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None, 'code': 'credit_balance_exhausted'}}" <br>

# step 4: chain <br>
### create 4 files and execute each such as $ python sequential_chain.py <br>
sequential_chain.py <br>
parallel_chains.py <br>
dynamic_routing.py <br>
advanced_lcel.py <br>

# step 5: build contexts <br>
### create 2 files and execute each such as $ python memory_fundamentals.py <br>
memory_fundamentals.py <br>
advanced_memory.py <br>

# step 6: build RAG <br>
### create 2 files and execute each such as $ python document_loader.py <br>
document_loader.py <br>
retrieval_chain.py <br>
