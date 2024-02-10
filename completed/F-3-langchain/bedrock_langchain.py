import os
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

##################################################################

llm = Bedrock(
    model_id="anthropic.claude-v2" #set the foundation model
)

##################################################################

prompt = "What is the largest city in Vermont?"
response_text = llm.invoke(prompt) #return a response to the prompt

print(response_text)
