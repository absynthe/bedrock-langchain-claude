import os
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_text_response(input_content): #text-to-text client function

    llm = Bedrock( #create a Bedrock llm client
        model_id="anthropic.claude-v2", #set the foundation model
        model_kwargs={
            "max_tokens_to_sample": 2000,
            "temperature": 0.5, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }
    )

    return llm.invoke(input_content) #return a response to the prompt
