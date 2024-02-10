import os
from langchain.llms.bedrock import Bedrock
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.json_loader import JSONLoader

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


#function to identify the metadata to capture in the vectorstore and return along with the matched content
def item_metadata_func(record: dict, metadata: dict) -> dict: 

    metadata["name"] = record.get("name")
    metadata["url"] = record.get("url")

    return metadata
    
    
def get_index(): #creates and returns an in-memory vector store to be used in the application
    
    embeddings = BedrockEmbeddings() #create a Titan Embeddings client
    
    loader = JSONLoader(
        file_path="services.json",
        jq_schema='.[]',
        content_key='description',
        metadata_func=item_metadata_func)

    text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
        separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
        chunk_size=8000, #based on this content, we just want the whole item so no chunking - this could lead to an error if the content is too long
        chunk_overlap=0 #number of characters that can overlap with previous chunk
    )
    
    index_creator = VectorstoreIndexCreator( #create a vector store factory
        vectorstore_cls=FAISS, #use an in-memory vector store for demo purposes
        embedding=embeddings, #use Titan embeddings
        text_splitter=text_splitter, #use the recursive text splitter
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #create an vector store index from the loaded PDF
    
    return index_from_loader #return the index to be cached by the client app
    

def get_similarity_search_results(index, question):
    raw_results = index.vectorstore.similarity_search_with_score(question)
    
    llm = get_llm()
    
    results = []
    
    for res in raw_results:
        content = res[0].page_content
        prompt = f"{content}\n\nSummarize how the above service addresses the following needs : {question}"
        
        summary = llm(prompt)
        
        results.append({"name": res[0].metadata["name"], "url": res[0].metadata["url"], "summary": summary, "original": content})
    
    return results

