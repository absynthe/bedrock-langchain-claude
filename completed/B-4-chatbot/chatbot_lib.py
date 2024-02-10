import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain

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


def get_memory(): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm()
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024) #Maintains a summary of previous messages
    
    return memory


def get_chat_response(input_text, memory): #chat client function
    
    llm = get_llm()
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model
    
    return chat_response
