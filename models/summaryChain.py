import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from userProfile import user_profile
#from articles import article

load_dotenv()

# create a template
summ_template = """
    Read the following article:
    {article}
    ---
    Make a bullet point list of the key ideas from the article that are relevant to this type of person:
    {user_profile}
    ---
"""

# assign template to prompt
summ_prompt = ChatPromptTemplate.from_template(summ_template)

# declare a model
summ_model = (
    ChatOpenAI(temperature=0.5)
)

# declare an output parser
summ_output_parser = StrOutputParser()

# setup chain
summ_chain = (
    {"article": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
    | summ_prompt
    | summ_model
    | summ_output_parser
)