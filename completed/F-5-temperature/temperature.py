import sys
import os
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_text_response(input_content, temperature): #text-to-text client function
    
    model_kwargs = {
            "max_tokens_to_sample": 512,
            "temperature": temperature, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }
    
    #create a Bedrock llm client
    llm = Bedrock(
        model_id="anthropic.claude-v2:1", #use the requested model
        model_kwargs = model_kwargs
    )
    
    return llm.invoke(input_content) #return a response to the prompt


for i in range(3):
    response = get_text_response(sys.argv[1], float(sys.argv[2]))
    print(response)

#Example 1: python temperature.py "Write a haiku about a long journey:" 0
#Example 2: python temperature.py "Write a haiku about a long journey:" 1