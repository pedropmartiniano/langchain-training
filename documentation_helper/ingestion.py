import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.environ.get('GOOGLE_API_KEY'),
    model='models/text-embedding-004'
)

def ingest_docs():
    loader = ReadTheDocsLoader('langchain-docs/api.python.langchain.com/en/latest')
    
    raw_documents = loader.load()
    print(f'loaded {len(raw_documents)} documents')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata['source']
        new_url = new_url.replace('langchain-docs', 'https:/')
        doc.metadata.update({'source': new_url})
        
    print(f'Going to add {len(documents)} documents to the Pinecone')
    
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=os.environ.get('INDEX_NAME'),
    )
    print('** Loading the documents to the Pinecone is done **')

if __name__ == "__main__":
    ingest_docs()