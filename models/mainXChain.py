from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
from langchain.chains import SequentialChain
from summaryChain import summ_chain
from ideaGenChain import idea_chain
from tweetWriterChain import tweet_chain
from userProfile import user_profile
from articles import article
import uvicorn

# Assuming your environment and chains are already set up correctly
load_dotenv()

# Define Pydantic models for input and output
class ChainInput(BaseModel):
    article: str
    user_profile: str

class ChainOutput(BaseModel):
    summary: str
    tweetIdeas: List[str]
    tweetDrafts: List[str]

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server for processing articles to tweets",
)
# please explain what this code does

# Create the chain instance outside of the route to avoid reinitialization on each request
overall_chain = SequentialChain(
    chains=[summ_chain, idea_chain, tweet_chain],
    input_variables=["article", "user_profile"],
    output_variables=["summary", "tweetIdeas", "tweetDrafts"],
    verbose=True
)


@app.post("/process-article", response_model=ChainOutput)
async def process_article(input: ChainInput):
    # Invoke chain with the input from the request
    result = overall_chain.invoke(
        {
            "article": input.article,
            "user_profile": input.user_profile
        }
    )
    return ChainOutput(**result)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



