# Open-Source-LLM

## LLaMA-2 Document-Based Q&A System
This project sets up a document-based Q&A system using LLaMA-2, LlamaIndex, and LangChain. It allows users to query a collection of documents and receive AI-generated responses based on the retrieved context.

### Features
  1) Uses LLaMA-2 (7B) from Hugging Face for natural language responses.
  2) Embeds and indexes documents using sentence-transformers/all-mpnet-base-v2.
  3) Optimized for GPU with 8-bit quantization (bitsandbytes).
  4) Retrieval-Augmented Generation (RAG) for accurate and contextual answers.

### Installation
    Run the following commands to install dependencies:
      pip install pypdf
      pip install -q transformers einops accelerate langchain bitsandbytes langchain_community
      pip install sentence_transformers
      pip install llama-index==0.9.39

### Setup
1. Prepare Document Storage
    Create a folder named Data and place your documents inside it:
    mkdir Data
2. Login to Hugging Face
    To access the LLaMA-2 model, authenticate with Hugging Face:
    huggingface-cli login

Run the Python script to load documents, create embeddings, and query the AI:

    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms import HuggingFaceLLM
    from llama_index.prompts.prompts import SimpleInputPrompt
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index.embeddings import LangchainEmbedding
    import torch

### Load documents
    documents = SimpleDirectoryReader("/content/Data/").load_data()

### Define system prompt
    system_prompt = """You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."""

### Query wrapper prompt
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

### Load LLaMA-2 model
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )

### Load embedding model
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

### Create service context
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

### Build vector index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

### Create query engine
    query_engine = index.as_query_engine()

### Run a query
    response = query_engine.query("What is a decoder?")
    print(response)

### Expected Output
    A decoder is a neural network module that transforms encoded input representations into human-readable text. It is commonly used in sequence-to-sequence models like transformers.

## Use Cases
    1) AI-powered chatbots trained on custom documents.
    2) Knowledge base search for enterprises.
    3) Automated document summarization and Q&A.

