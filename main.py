import os
from icecream import ic
from typing import Literal

from langchain_openai.chat_models import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatOpenAI(
    model="gpt-5-mini",
    reasoning_effort="low"
)

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """# あなた
「あなたは熟練したリサーチャーです。あなたの仕事は徹底的に調査を行い、その後、洗練されたレポートを作成することです。

あなたはいくつかのツールにアクセスできます。

## `internet_search`

これを使用して、指定したクエリでインターネット検索を実行します。結果数、トピック、および生のコンテンツを含めるかどうかを指定できます。」

"""

# Create the agent
agent = create_deep_agent(
    [internet_search],
    research_instructions,
    model=model,
)

# Invoke the agent
result = agent.invoke({"messages": [{"role": "user", "content": "新潟県の天気を調べて"}]})

ic(result)