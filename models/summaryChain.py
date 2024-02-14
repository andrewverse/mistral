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
    Carefully read the article below and distill its essence into key points. Consider the interests and background of the reader profile provided, highlighting aspects of the article they would find most compelling and useful.
    
    Article:
    {article}
    
    Reader Profile:
    {user_profile}
    
    Instructions:
    - Extract the core message and key takeaways.
    - Present them as concise, bullet-pointed highlights.
    - Tailor the summary to resonate with the reader's specific interests and knowledge level.
    
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