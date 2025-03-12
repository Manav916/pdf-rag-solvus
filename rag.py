import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

client = openai.OpenAI()

def process_chunks(result):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(result)
    chunk_summaries = []
    for chunk in chunks:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Provide a brief summary of the following text passage. Mention key points and important details:"},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.2,
                max_tokens=150
            )
            summary = response.choices[0].message.content.strip()
            # print(f"Chunk summary: {summary}")
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")
            continue
        chunk_summaries.append(summary)

    docs = [{"id": f"{i}", "text": chunk} for i, chunk in enumerate(chunk_summaries)]
    return docs

def create_vector_db(docs):
    client = chromadb.Client(Settings())
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key, model_name="text-embedding-ada-002"
    )
    collection = client.get_or_create_collection(
        name="collection_magi", embedding_function=embedding_function
    )
    existing_ids = set(collection.get().get("ids", []))

    for doc in docs:
        if doc["id"] not in existing_ids:
            collection.add(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[{"source": doc["id"]}]
            )
    return collection

def retrieve_relevant_docs(query, k=5, collection=None):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=['documents', 'distances']
    )
    
    threshold = 0.8
    filtered_docs = [
        doc for doc, dist in zip(results["documents"][0], results["distances"][0]) 
        if dist < threshold
    ]
    results["documents"][0] = filtered_docs if filtered_docs else results["documents"][0][:1]
    retrieved_docs = results["documents"][0]
    return retrieved_docs

def query_llm(query, context_docs):
    context = "\n".join(context_docs)
    prompt = (
        "You need to answer questions regarding a short story 'The Gift of the Magi' by O. Henry. Context will be provided along with the query. You need to form an answer based on the context provided."
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Do not hallucinate. Keep all your answers grounded to the context provided."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    answer = response.choices[0].message.content.strip()    
    return answer
