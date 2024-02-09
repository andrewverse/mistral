from dotenv import load_dotenv
from langchain.chains import SequentialChain
from summaryChain import summ_chain
from ideaGenChain import idea_chain
from tweetWriterChain import tweet_chain
from userProfile import user_profile
from articles import article

load_dotenv()

# create chain
overall_chain = SequentialChain(
    chains=[summ_chain, idea_chain, tweet_chain],
    input_variables=["article", "user_profile"],
    output_variables=["summary", "tweetIdeas","tweetDrafts"],
    verbose=True
)

# invoke chain
articleToTweets = overall_chain.invoke(
    {
        "article": article,
        "user_profile": user_profile
    }
)

print(articleToTweets)