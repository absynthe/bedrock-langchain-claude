import os
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

embeddings = BedrockEmbeddings() #create a Titan Embeddings client

def get_index(): #creates and returns an in-memory vector store to be used in the application
    loader = CSVLoader(file_path="sagemaker_answers.csv")

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
    )

    index_from_loader = index_creator.from_loaders([loader])
    
    return index_from_loader
    

def get_similarity_search_results(index, question):
    results = index.vectorstore.similarity_search_with_score(question)
    
    flattened_results = [{"content":res[0].page_content, "score":res[1]} for res in results] #flatten results for easier display and handling
    
    return flattened_results


def get_embedding(text):
    return embeddings.embed_query(text)