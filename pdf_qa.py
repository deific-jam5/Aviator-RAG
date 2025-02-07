import os
import ollama
import json
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

# ---------------------- CONFIGURATION ----------------------
PDF_DIRECTORY = "pdfs"  # Folder where PDFs are stored
DB_DIRECTORY = "vector_db"  # Where to store vector embeddings
MODEL_NAME = "llama2"  # LLaMA 2 model in Ollama

# ---------------------- STEP 1: Extract Text from PDFs ----------------------
def extract_text_from_pdfs():
    """Loads all PDFs from the directory and extracts text."""
    documents = []
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        print(f"📁 Created '{PDF_DIRECTORY}' folder. Place your PDFs inside.")
        return []

    for file in os.listdir(PDF_DIRECTORY):
        if file.endswith(".pdf"):
            print(f"📄 Loading: {file}")
            try:
                loader = PyPDFLoader(os.path.join(PDF_DIRECTORY, file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue

    if not documents:
        print("❌ No PDFs found! Please add some files to the 'pdfs' folder.")
        return []
    
    print(f"✅ Extracted {len(documents)} pages from PDFs")
    return documents

# ---------------------- STEP 2: Split Text into Chunks ----------------------
def split_text_chunks(documents):
    """Splits the extracted text into smaller chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} text chunks")
    return chunks

# ---------------------- STEP 3: Store Data in Vector Database ----------------------
def store_in_vector_db(chunks):
    """Converts text chunks into embeddings and stores them in a vector database."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Create the vector store and persist it to the specified directory
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model, 
            persist_directory=DB_DIRECTORY
        )
        print("✅ PDF data stored in vector database!")
        return vector_db
    except Exception as e:
        print(f"❌ Error storing data in vector database: {e}")
        return None

# ---------------------- STEP 4: Load Vector Database ----------------------
def load_vector_db():
    """Loads the vector database for querying."""
    if not os.path.exists(DB_DIRECTORY) or not os.listdir(DB_DIRECTORY):
        print("❌ Vector database not found or is empty. Please process PDFs first.")
        return None

    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embedding_model)
        # Create a retriever from the vector store
        retriever = vector_db.as_retriever()
        return retriever
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return None

# ---------------------- STEP 5: Query LLaMA 2 ----------------------
def ask_llama(question, retriever):
    """Queries the stored vector database and streams responses from LLaMA 2."""
    try:
        llm = OllamaLLM(model=MODEL_NAME)

        # Define a prompt template
        prompt_template = PromptTemplate.from_template(
            "Answer the following question based on the context:\n\nQuestion: {question}\n\nContext: {context}"
        )

        # Define a Runnable sequence
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # Pass the question and retrieve context
            | prompt_template  # Format the input using the prompt template
            | llm  # Pass the formatted input to the LLM
        )

        # Stream the response token by token
        print("🤖 LLaMA: ", end="", flush=True)  # Print the bot's prefix
        full_response = ""
        for chunk in qa_chain.stream(question):  # Stream the response
            print(chunk, end="", flush=True)  # Print each chunk as it arrives
            full_response += chunk  # Accumulate the full response

        print("\n")  # Add a newline after the response
        return full_response  # Return the full response (optional)
    except Exception as e:
        print(f"❌ Error querying LLaMA 2: {e}")
        return "Sorry, I encountered an error while processing your question."

# ---------------------- MAIN FUNCTION ----------------------
if __name__ == "__main__":
    # Check if vector DB exists
    if not os.path.exists(DB_DIRECTORY):
        print("⚡ Processing PDFs for the first time...")
        docs = extract_text_from_pdfs()
        if docs:
            chunks = split_text_chunks(docs)
            vector_db = store_in_vector_db(chunks)
        else:
            print("❌ No PDFs to process. Exiting...")
            exit()
    else:
        print("🔍 Using existing vector database.")

    # Load vector DB and create a retriever
    retriever = load_vector_db()
    if retriever is None:
        print("❌ Failed to load vector database. Exiting...")
        exit()

    # Interactive mode
    print("\n💬 Type your question (or type 'exit' to quit)")
    while True:
        user_input = input("📝 You: ")
        if user_input.lower() == "exit":
            print("👋 Exiting chat.")
            break

        response = ask_llama(user_input, retriever)
        print(f"🤖 LLaMA: {response}\n")
