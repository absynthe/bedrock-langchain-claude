import sys
import os
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_inference_parameters(model): #return a default set of parameters based on the model's provider
    bedrock_model_provider = model.split('.')[0] #grab the model provider from the first part of the model id

    if (bedrock_model_provider == 'anthropic'): #Anthropic model
        return {
            "max_tokens_to_sample": 512,
            "temperature": 1, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }
    
    elif (bedrock_model_provider == 'ai21'): #AI21
        return { 
            "maxTokens": 512, 
            "temperature": 0, 
            "topP": 0.5, 
            "stopSequences": [], 
            "countPenalty": {"scale": 0 }, 
            "presencePenalty": {"scale": 0 }, 
            "frequencyPenalty": {"scale": 0 } 
           }
    
    elif (bedrock_model_provider == 'cohere'): #COHERE
        return {
            "max_tokens": 512,
            "temperature": 0,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
    
    elif (bedrock_model_provider == 'meta'): #META
        return {
            "temperature": 0,
            "top_p": 0.9,
            "max_gen_len": 512
        }

    else: #Amazon
        #For the LangChain Bedrock implementation, these parameters will be added to the
        #textGenerationConfig item that LangChain creates for us
        return { 
            "maxTokenCount": 512, 
            "stopSequences": [], 
            "temperature": 1, 
            "topP": 0.999 
        }
    

#

def get_text_response(model, input_content): #text-to-text client function
    
    model_kwargs = get_inference_parameters(model) #get the default parameters based on the selected model
    
    #create a Bedrock llm client
    llm = Bedrock(
        model_id=model, #use the requested model
        model_kwargs = model_kwargs
    )
    
    return llm.invoke(input_content) #return a response to the prompt

response = get_text_response(sys.argv[1], sys.argv[2])

#Example: 
# Argument 1: anthropic.claude-v2:1
# Argument 2: "Write a Haiku poem about Amazon Web Services." -> Keep the quotes, otherwise the model will only see the first word. 

print(response)