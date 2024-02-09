from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from userProfile import user_profile
from summaryChain import summary

load_dotenv()

# create a template
idea_template = """
    Generate a list of 10 tweet ideas based on the following article summary.
    {summary}
    ---
    Make each tweet under 280 characters long.
    ---
    Keep in the tweets relevant to the following user profile:
    {user_profile}
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

# invoke chain
tweetIdeas = idea_chain.invoke(
    {
        "summary": summary,
        "user_profile": user_profile
    }
)

print(tweetIdeas)