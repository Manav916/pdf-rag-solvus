import os
from dotenv import load_dotenv
from preprocess import process_document
from ocr import perform_ocr
from rag import process_chunks, create_vector_db, retrieve_relevant_docs, query_llm
from summarize import summarize
import streamlit as st

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="The Gift of the Magi - PDF Analysis",
    page_icon="📚",
    layout="wide"
)
st.cache_resource.clear()


@st.cache_data
def process_pdf(pdf_path):
    """Process the PDF and return the extracted text"""
    try:
        image_urls = process_document(pdf_path)
        ocr_results = perform_ocr(image_urls)
        ocr_content = [ocr_result['ocr_content'] for ocr_result in ocr_results]
        result = " ".join(ocr_content)
        return result
    except Exception:
        return None

@st.cache_data
def prepare_docs(text):
    """Prepare the documents for the RAG system"""
    try:
        docs = process_chunks(text)
        return docs
    except Exception:
        return None

@st.cache_resource
def prepare_rag_system(docs):
    """Prepare the RAG system with the processed text"""
    try:
        collection = create_vector_db(docs)
        return collection
    except Exception:
        return None

def main():
    st.title("The Gift of the Magi - PDF Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.radio(
        "Choose an option:",
        ["Summary", "Ask Questions (RAG)"]
    )
    
    # Define the PDF path
    pdf_path = "assets/The_Gift_of_the_Magi.pdf"
    
    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        st.error(f"Error: The PDF file '{pdf_path}' does not exist.")
        return
    
    # Process the PDF with a spinner outside the cached function
    with st.spinner("Processing PDF..."):
        result = process_pdf(pdf_path)
    
    if result is None:
        st.error("Failed to process the PDF. Please check the logs for more details.")
        return
    
    # Display based on the selected option
    if option == "Summary":
        st.header("Summary of 'The Gift of the Magi'")
        
        if st.button("Generate Summary"):
            try:
                with st.spinner("Generating summary..."):
                    summary_text = summarize(result)
                    st.write(summary_text)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
    
    else:  # RAG option
        st.header("Ask Questions about 'The Gift of the Magi'")
        
        # Prepare RAG system with a spinner outside the cached function
        with st.spinner("Setting up RAG system..."):
            docs = prepare_docs(result)
            collection = prepare_rag_system(docs)
        
        if collection is None:
            st.error("Failed to set up the RAG system. Please check the logs for more details.")
            return
        
        # User query input
        query = st.text_input("Enter your question:", placeholder="How much salary was Jim earning?")
        
        if query:
            if st.button("Get Answer"):
                try:
                    with st.spinner("Retrieving answer..."):
                        # Retrieve relevant documents
                        relevant_docs = retrieve_relevant_docs(query, 3, collection)
                        
                        # Query the LLM
                        llm_response = query_llm(query, relevant_docs)
                        
                        # Display the answer
                        st.subheader("Answer:")
                        st.write(llm_response)
                        
                        # Optionally show the relevant context
                        with st.expander("View relevant context"):
                            for i, doc in enumerate(relevant_docs):
                                st.write(f"**Context {i+1}:**")
                                st.write(doc)
                                st.write("---")
                except Exception as e:
                    st.error(f"Error retrieving answer: {str(e)}")

if __name__ == "__main__":
    main()

