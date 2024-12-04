import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

if __name__ == '__main__':
    pdf_path = '/home/pmartiniano/projects/side-projects/langchain-course/vector_in_memory/react.pdf'
    loader = PyPDFLoader(pdf_path)
    # Load the document from the PDF, that returns a list of Documents
    document = loader.load()
    
    # creating the text splitter object to split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    
    # Split the document into chunks
    docs = text_splitter.split_documents(documents=document)

    # Create the embeddings object
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        model='models/text-embedding-004'
    )
    
    # creating the vectors from the embeddings docs
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the vectorstore locally
    vectorstore.save_local('faiss_index_react')
    
    # Load the vectorstore from the local file
    new_vectorstore = FAISS.load_local(
        'faiss_index_react',
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Geting the RAG prompt from the hub
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0
    )
    # Creating the chain
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt
    )
    
    # Creating the retrieval chain
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),
        combine_docs_chain
    )
    
    # Invoking the chain
    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res['answer'])
    
    print('-')