import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from userProfile import user_profile
#from summaryChain import summary

load_dotenv()

# create a template
idea_template = """
    Utilizing the article summary provided, craft 10 creative and engaging brief insights. 
    These brief insights should encapsulate the article's insights and be tailored to the reader's interests and background. 
    Remember, each breif insight should stand on its own as a compelling piece of writing.

    Article Summary:
    {summary}
    
    Reader Profile:
    {user_profile}
    
    Instructions:
    - Ensure each brief insight is succinct, not exceeding 500 characters.
    - Focus on sparking curiosity and providing value.
    - Infuse a tone that aligns with the reader's perspective and interests.
    
    ---
"""

# assign template to prompt
idea_prompt = ChatPromptTemplate.from_template(idea_template)

# declare a model
idea_model = (
    ChatOpenAI(temperature=0.5)
)

# declare an output parser
idea_output_parser = StrOutputParser()

# setup chain
idea_chain = (
    {"summary": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
    | idea_prompt
    | idea_model
    | idea_output_parser
)