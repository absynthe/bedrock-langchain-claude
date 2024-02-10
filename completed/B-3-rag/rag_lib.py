import os
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_llm():

    model_kwargs = {
            "max_tokens_to_sample": 2000,
            "temperature": 0.5, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }

    llm = Bedrock(
        model_id="anthropic.claude-v2", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
 
    return llm


def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings() #create a Titan Embeddings client
    
    pdf_path = "2022-Shareholder-Letter.pdf" #assumes local PDF file with this name

    loader = PyPDFLoader(file_path=pdf_path) #load the pdf file
    
    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=1000, #divide into 1000-character chunks using the separators above
        chunk_overlap=100 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF
    
    return index_from_loader #return the index to be cached by the client app


def get_rag_response(index, question): #rag client function
    
    llm = get_llm()
    
    response_text = index.query(question=question, llm=llm) #search against the in-memory index, stuff results into a prompt and send to the llm
    
    return response_text