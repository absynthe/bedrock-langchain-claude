import os
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

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

#

def get_prompt(adjective, noun, verb):
    
    template = "Tell me a story about a {adjective} {noun} who loves to {verb}:"

    prompt_template = PromptTemplate.from_template(template) #this will automatically identify the input variables for the template

    prompt = prompt_template.format(noun=noun, adjective=adjective, verb=verb)
    
    return prompt

#

def get_text_response(adjective, noun, verb): #text-to-text client function
    llm = get_llm()
    
    prompt = get_prompt(adjective, noun, verb)
    
    return llm.invoke(prompt) #return a response to the prompt

#

