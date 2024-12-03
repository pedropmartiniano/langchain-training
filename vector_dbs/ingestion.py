import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
    print('ingesting...') 
    # Loading the document
    loader = TextLoader("/home/pmartiniano/projects/side-projects/langchain-course/vector_dbs/mediumblog1.txt")
    document = loader.load()

    print('splitting...')
    # Creating the text splitter with the desired chunk size and overlap
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Splitting the original document into chunks(smaller documents)
    texts = text_splitter.split_documents(document)
    
    # Creating the embeddings object with the Google API key and the model name
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.environ.get('GOOGLE_API_KEY'), model='models/text-embedding-004')
    
    print('ingesting...')
    # Generating vector embeddings for each chunk
    # Ingesting the vector chunks into the Pinecone vector store
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get('INDEX_NAME'))
    print('finish')