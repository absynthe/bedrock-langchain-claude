import os
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyPDFLoader

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


pdf_path = "uploaded_file.pdf"

def get_example_file_bytes(): #provide the file bytes so the user can download a ready-made example
    with open("2022-Shareholder-Letter.pdf", "rb") as file:
        file_bytes = file.read()
    
    return file_bytes


def save_file(file_bytes): #save the uploaded file to disk to summarize later
    
    with open(pdf_path, "wb") as f: 
        f.write(file_bytes)
    
    return f"Saved {pdf_path}"


def get_docs():
    
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], chunk_size=4000, chunk_overlap=100 
    )
    docs = text_splitter.split_documents(documents=documents)
    
    return docs

def get_summary(return_intermediate_steps=False):
    
    map_prompt_template = "{text}\n\nWrite a few sentences summarizing the above:"
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    
    combine_prompt_template = "{text}\n\nWrite a detailed analysis of the above:"
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
    
    
    llm = get_llm()
    docs = get_docs()
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, return_intermediate_steps=return_intermediate_steps)
    
    if return_intermediate_steps:
        return chain({"input_documents": docs}, return_only_outputs=True) #make return structure consistent with chain.run(docs)
    else:
        return chain.run(docs)
