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
    Examine the provided short-form content ideas, identifying the three with the greatest potential for impact and engagement. 
    Refine these insights to heighten their clarity and resonance. 
    Ensure they are primed for social media interaction, mirroring the interests and expectations of the targeted reader profile.
    Keep in mind that each insight should be a standalone piece of content, not exceeding 280 characters.

    Provided Content Ideas:
    {tweetIdeas}
    
    Targeted Reader Profile:
    {user_profile}
    
    Enhancement Guidelines:
    - Use language that elevates the content, making it more impactful and memorable.
    - Clarify the core message of each insight to ensure immediate understanding.
    - Tailor the expression and tone to match the reader's preferences and expectations.
    - Ensure each of the three final pieces of content is less than 280 characters.
    
    ---
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