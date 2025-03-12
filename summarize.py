from langchain.chat_models import init_chat_model
import tiktoken
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain

# Convert text chunks to Document objects
# docs = [Document(page_content=text) for text in result]

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: gpt-3.5-turbo)
                     For LLaMA models, you can use "cl100k_base" encoding
    
    Returns:
        int: The number of tokens in the text
    """
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    return len(tokens)

def summarize(result):
    llm = init_chat_model("llama3-70b-8192", model_provider="groq")
    
    prompt_text = "Provide a comprehensive summary for the given story. The summary should cover all the key points and main ideas presented by the author in the story, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the characters and plot, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. Text:\n\n" + result
    
    # Count tokens in the prompt
    token_count = count_tokens(prompt_text, "cl100k_base")  # Using cl100k_base for LLaMA compatibility
    if token_count < 4000:
    
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a comprehensive summary for the given story. The summary should cover all the key points and main ideas presented by the author in the story, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the characters and plot, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. Text:\\n\\n{context}")
        ])

        chain = create_stuff_documents_chain(llm, prompt)
        summary = chain.invoke({"context": [Document(page_content=result)]})
        return summary
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        docs = text_splitter.create_documents([result])

        # Create map and reduce prompts
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following text:\n\n{context}")
        ])
        
        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", """Combine these summaries into a single coherent summary that captures the key points and main themes:
            
            {context}""")
        ])

        # Create map and reduce chains
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        # Create and run map reduce chain
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_chain,
            document_variable_name="context"
        )

        summary = map_reduce_chain.invoke({"input_documents": docs})
        return summary["output_text"]