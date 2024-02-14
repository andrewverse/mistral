from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
from langchain.chains import SequentialChain
from summaryChain import summ_chain
from ideaGenChain import idea_chain
from tweetWriterChain import tweet_chain
#from userProfile import user_profile
#from articles import article
from langchain_core.runnables import RunnablePassthrough
import uvicorn

# Assuming your environment and chains are already set up correctly
load_dotenv()

# Define Pydantic models for input and output
class ChainInput(BaseModel):
    article: str
    user_profile: str

class ChainOutput(BaseModel):
    tweetDrafts: str

# Create FastAPI app to set up the API server
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server for processing articles to tweets",
)

# Create overall chain of to link summarization, idea generation, and tweet writing chains
## Create the chain instance outside of the route to avoid reinitialization on each request
overall_chain = (
    {"article": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
    | summ_chain 
    | idea_chain 
    | tweet_chain
)

# This route will take an article and user profile, and return the summary, tweet ideas, and tweet drafts
@app.post("/process-article", response_model=ChainOutput)
async def process_article(input: ChainInput):
    """Generate tweet drafts based on article and user profile.

    Args:
        input (ChainInput): Contains article and user profile.

    Returns:
        ChainOutput: Populated with tweet drafts.
    """
    
    # Invoke LangChain overall_chain
    result = overall_chain.invoke(
        {
            "article": input.article,
            "user_profile": input.user_profile
        }
    )
    
    return ChainOutput(tweetDrafts=result)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)