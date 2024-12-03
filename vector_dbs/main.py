import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # Query without the contextualization of Pinecone
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model='gemini-1.5-flash'
    )
    
    query = "What is Pinecone in machine learning?"
    
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)
    
    print('-----------------')

    # Query with the contextualization of Pinecone
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get('GOOGLE_API_KEY'),
        model='models/text-embedding-004'
    )
    
    vectorstore = PineconeVectorStore(index_name=os.environ.get('INDEX_NAME'), embedding=embeddings)
    
    retrieval_qa_chat_prompt = hub.pull('langchain-ai/retrieval-qa-chat')
    
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )
    
    result = retrival_chain.invoke(input={"input": query})
    print(result)
    
    print('-----------------')
    
    # Customized RAG Prompt with context of Pinecone vector database
    template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences and keep the answer as concise as possible.
        Always say "Thanks for asking!" at the end of the answer.
        
        {context}
        
        Question: {question}
        
        Helpful Answer:
    """
    
    custom_rag_prompt = PromptTemplate.from_template(template=template)
    
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        } |
        custom_rag_prompt |
        llm
    )
    
    res = rag_chain.invoke(query)
    print(res)