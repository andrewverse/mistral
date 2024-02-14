import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from userProfile import user_profile
#from ideaGenChain import tweetIdeas

load_dotenv()

# create a template
tweet_template = """
    Choose the 3 most compelling tweet ideas from the following list:
    {tweetIdeas}
    ---
    Revise them for clarity and impact.
    ---
    Keep the tweets relevant to the following user profile:
    {user_profile}
"""

# assign template to prompt
tweet_prompt = ChatPromptTemplate.from_template(tweet_template)

# declare a model
tweet_model = (
    ChatOpenAI(temperature=0.5)
)

# declare an output parser
tweet_output_parser = StrOutputParser()

# setup chain
tweet_chain = (
    {"tweetIdeas": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
    | tweet_prompt
    | tweet_model
    | tweet_output_parser
)