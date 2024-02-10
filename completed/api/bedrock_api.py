import os
import json
import boto3

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

session = boto3.Session() 

#creates a Bedrock client
bedrock = session.client(
    service_name='bedrock-runtime', 
) 

#set the foundation model
bedrock_model_id = "anthropic.claude-v2" 

#the prompt to send to the model
prompt = """
Human: What is the largest city in New Hampshire?
Assistant:
"""

#build the request payload for Claude
body = json.dumps({
    "prompt":prompt,
    "max_tokens_to_sample":2000,
    "temperature":0,
    "top_k":250,
    "top_p":0.9,
    "stop_sequences":["\n\nHuman:"]
    }) 

#send the payload to Bedrock
response = bedrock.invoke_model(
    body=body,
    modelId=bedrock_model_id,
    accept='application/json',contentType='application/json'
    )

#read the response
response_body = json.loads(
    response.get('body').read()
    )

#extract the text from the JSON response
response_text = response_body.get("completion") 

print(response_text)
