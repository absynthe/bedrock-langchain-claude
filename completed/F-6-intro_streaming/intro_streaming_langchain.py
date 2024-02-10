import os
import json
import boto3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

session = boto3.Session() #sets the profile name to use for AWS credentials

model_kwargs = {
    "max_tokens_to_sample": 512,
    "temperature": 0.5, 
    "top_k": 250, 
    "top_p": 0.999, 
    "stop_sequences": ["\n\nHuman:"] 
    }

#create a Bedrock llm client
llm = Bedrock(
    model_id="anthropic.claude-v2:1", #use the requested model
    model_kwargs = model_kwargs,
    streaming=True, #enable streaming mode
    callbacks=[StreamingStdOutCallbackHandler()]
    )

prompt = "\n\nHuman:Tell me a story about two puppies and two kittens who became best friends\n\nAssistant:"
                
llm.invoke(prompt)

