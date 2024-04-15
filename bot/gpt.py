from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv(override=True)

tools = [TavilySearchResults(max_results=1)]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Only use the tavily_search_results_json tool for up to date information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
resp = agent_executor.invoke({"input": "What is the best food to eat in Jakarta as of 2023?"})
print(resp.get("output"))

