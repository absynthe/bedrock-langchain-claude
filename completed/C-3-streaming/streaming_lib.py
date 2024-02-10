import os
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock

#sets the profile name to use for AWS credentials
os.environ["AWS_PROFILE"] = "bedrock-ana" #replace with your profile name

def get_llm(streaming_callback):

    model_kwargs = {
            "max_tokens_to_sample": 2000,
            "temperature": 0.5, 
            "top_k": 250, 
            "top_p": 0.999, 
            "stop_sequences": ["\n\nHuman:"] 
           }

    llm = Bedrock(
        model_id="anthropic.claude-v2", #set the foundation model
        model_kwargs=model_kwargs,
        streaming=True,
        callbacks=[streaming_callback]
        ) 
 
    return llm


def get_streaming_response(prompt, streaming_callback):
    conversation_with_summary = ConversationChain(
        llm=get_llm(streaming_callback)
    )
    return conversation_with_summary.predict(input=prompt)
