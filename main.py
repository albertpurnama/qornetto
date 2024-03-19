import discord
from discord.ext import commands
import traceback

from dotenv import load_dotenv
load_dotenv(override=True)

import os
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", 'invalid_token')

from langchain_openai import ChatOpenAI, OpenAI
from langchain.memory import ConversationEntityMemory, UpstashRedisChatMessageHistory
from langchain.memory.entity import UpstashRedisEntityStore
from langchain_core.prompts.prompt import PromptTemplate
from typing import Any, Optional

chat = ChatOpenAI()

userMessageCounter = 2;

# Import the original class
from langchain.memory.entity import UpstashRedisEntityStore

# Create a subclass with the necessary fixes
# This is temporary fix until https://github.com/langchain-ai/langchain/pull/18892
# is merged.
class FixedUpstashRedisEntityStore(UpstashRedisEntityStore):
    redis_client: Any
    session_id: str = "default"
    key_prefix: str = "memory_store"
    ttl: Optional[int] = 60 * 60 * 24
    recall_ttl: Optional[int] = 60 * 60 * 24 * 3

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
      ):
        super().__init__(*args, **kwargs)

entity_store = FixedUpstashRedisEntityStore(
    session_id="my-session",
    url="https://enabling-anemone-37187.upstash.io",
    # TODO: move this to env variables
    token=os.getenv("UPSTASH_REDIS_TOKEN", "invalid-upstash-redis-token"),
    ttl=600,
)

llm = OpenAI(temperature=0)
memory = ConversationEntityMemory(llm=llm, chat_history_key="history")
memory.entity_store = entity_store

# from langchain.globals import set_debug
# set_debug(True)


_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = """You Qornet, you are an assistant to a human, powered by a large language model trained by OpenAI.

You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate dog-like text based on the input you receive, allowing you to engage in dog-human conversations and provide responses that are coherent and relevant to the topic at hand.

You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.

Context:
{entities}

Current conversation:
{history}
"""

intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    botUser = bot.user
    name = 'UnknownBot'
    if botUser is not None:
        name = botUser.name
    print(f'Logged in as {name}')

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

@tool
def search(query: str) -> str:
    """Look up things online."""
    return "LangChain"

tools = [search]

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

chatPromptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["entities", "history"], template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE)),
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
    MessagesPlaceholder(variable_name='agent_scratchpad'),
])

# Construct the OpenAI Functions agent
agent = create_openai_functions_agent(llm, tools, prompt=chatPromptTemplate)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# send_message sends messsage to discord, but it also handles
# when the message is long by cutting it into 2000 characters
# messages. It should break the characters into word
# level break, rather than a character level break.
async def send_message(message:str, messageChannel):
    message_to_process = message

    while len(message_to_process) > 0:
        if len(message_to_process) > 2000:
            curr_message = message_to_process[0:2000];
            curr_message = curr_message.split(' ')
            poppedString = curr_message.pop()
            curr_message = ' '.join(curr_message)
            message_to_process = message_to_process[2000:]
            message_to_process = poppedString + message_to_process
            await messageChannel.send(curr_message)
        else:
            await messageChannel.send(message_to_process)
            message_to_process = ""

@bot.event
async def on_message(message):
    # check the message sender
    # if it's coming from Nonuts, do not reduce counter.
    if message.author.name != "Nonuts":
      global userMessageCounter
      userMessageCounter = userMessageCounter - 1;
    
    if (bot.user and bot.user.mentioned_in(message)) or userMessageCounter < 1: 
        userMessageCounter = 2;
        # Fetch last 5 messages

        rep = agent_executor.invoke({"input": message.content})

        try:
            await send_message(rep['output'], message.channel)
        except Exception as e:
            # log the error trace.
            traceback.print_exception(type(e), e, e.__traceback__)
            print("oops, something went wrong when sending the message to message channel");

    try:
        await bot.process_commands(message)
    except Exception as e:
        # log the error trace.
        traceback.print_exception(type(e), e, e.__traceback__)
        print("oops, something went wrong when processing the bot commands");

@bot.command()
async def hello(ctx):
    await ctx.send(f"Hello, {ctx.author.mention}!")

bot.run(BOT_TOKEN)
