#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive LlamaIndex RAG Demo
Users can input queries and get answers
"""

import os
import sys

# Set OpenAI API key (users need to fill in their own)
os.environ["OPENAI_API_KEY"] = ""  # Please enter your OpenAI API key here


def check_dependencies():
    """Check dependencies"""
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core import StorageContext
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.vector_stores.faiss import FaissVectorStore
        import faiss

        return True
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def setup_rag_system():
    """Setup RAG system"""
    print("Setting up RAG system...")

    # Basic settings
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI(model="gpt-4o")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Load documents - use simple PDF reader
    from llama_index.core import SimpleDirectoryReader

    try:
        local_loader = SimpleDirectoryReader(
            input_dir="./data",
            required_exts=[".pdf", ".docx", ".pptx", ".epub", ".md"],
        )

        documents = local_loader.load_data()
        print(f"✓ Loaded {len(documents)} documents")

        if not documents:
            print(
                "No documents found, please ensure there are PDF files in the data directory"
            )
            return None

    except Exception as e:
        print(f"Document loading failed: {e}")
        return None

    # Document chunking
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        paragraph_separator="\n\n", # make sure separator will not split table data
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✓ Document chunking completed, {len(nodes)} nodes total")

    # Create vector index
    import faiss
    from llama_index.core import StorageContext
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    d = 1536
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

    query_engine = RetrieverQueryEngine.from_args(retriever=vector_retriever)

    print("✓ RAG system setup completed")
    return query_engine


def interactive_query(query_engine):
    """Interactive query"""
    print("\n" + "=" * 50)
    print("Interactive RAG Query System")
    print("Enter 'quit' or 'exit' to exit")
    print("Enter 'debug' to view retrieval results of the last query")
    print("Enter 'debug <query>' to view retrieval results of a specific query")
    print("Note: You need to perform at least one query before using debug")
    print("=" * 50)

    last_query = None  # Record the last query

    while True:
        try:
            query = input("\nPlease enter your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if query.lower().startswith("debug"):
                # Show retrieved content
                from llama_index.core.retrievers import VectorIndexRetriever

                # Parse debug command
                if query.lower() == "debug":
                    # Use the last query
                    if not last_query:
                        print(
                            "\nNo previous query record, please perform a query first"
                        )
                        continue
                    debug_query = last_query
                else:
                    # Extract query after debug
                    debug_query = query[6:].strip()  # Remove "debug " prefix
                    if not debug_query:
                        if not last_query:
                            print(
                                "\nNo query content specified and no previous query record"
                            )
                            continue
                        debug_query = last_query

                print(f"\nUsing query: '{debug_query}' for debug")

                # look up the retrieval results of the query
                retrieved_nodes = query_engine.retriever.retrieve(debug_query)
                print(f"\nRetrieved {len(retrieved_nodes)} nodes:")
                for i, node in enumerate(retrieved_nodes):
                    print(f"\nNode {i + 1}:")
                    print(f"Similarity score: {node.score}")
                    print(f"Content: {node.node.text[:1000]}...")
                    print("-" * 50)
                continue

            if not query:
                continue

            print("\nQuerying...")
            response = query_engine.query(query)
            print(f"\nAnswer: {response}")

            # Record this query
            last_query = query

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Query error: {e}")


def main():
    """Main function"""
    print("LlamaIndex RAG Interactive Demo")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OpenAI API key not set")
        print("Please set your API key in the code or set environment variable")
        api_key = input(
            "Please enter your OpenAI API key (or press Enter to skip): "
        ).strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("API key not set, program may not work properly")

    # Setup RAG system
    query_engine = setup_rag_system()
    if query_engine is None:
        return

    # Start interactive query
    interactive_query(query_engine)


if __name__ == "__main__":
    main()
