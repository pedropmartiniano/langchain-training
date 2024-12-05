import os
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain import hub


load_dotenv()
INDEX_NAME = os.environ.get('INDEX_NAME')

def run_llm(query: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        model='models/text-embedding-004'
    )
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-pro',
        temperature=0,
        verbose=True
    )
    
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )
    
    result = qa.invoke(input={'input': query})
    return result

if __name__ == "__main__":
    query = 'What is a LangChain Chain?'
    result = run_llm(query)
    print(result['answer'])
