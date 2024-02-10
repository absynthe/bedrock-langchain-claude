import os
import json
from json import JSONDecodeError
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_llm():

    model_kwargs = {
            "max_tokens_to_sample": 2000,
            "temperature": 0.0, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }

    llm = Bedrock(
        model_id="anthropic.claude-v2", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
 
    return llm

def validate_and_return_json(response_text):
    try:
        response_json = json.loads(response_text) #attempt to load text into JSON
        return False, response_json, None #returns has_error, response_content, err 
    
    except JSONDecodeError as err:
        return True, response_text, err #returns has_error, response_content, err 

#

def get_json_response(input_content): #text-to-text client function
    
    llm = get_llm()

    response = llm.invoke(input_content) #the text response for the prompt
    
    return validate_and_return_json(response)
