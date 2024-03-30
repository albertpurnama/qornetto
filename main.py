import logging
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

userMessageCounter = 100;

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

    def clear(self):
        print("[FixedUpstashRedisEntityStore] Clear method called")
        super().clear()

entity_store = FixedUpstashRedisEntityStore(
    session_id="my-session",
    url="https://enabling-anemone-37187.upstash.io",
    token=os.getenv("UPSTASH_REDIS_TOKEN", "invalid-upstash-redis-token"),
    ttl=None,
)

_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = """You are Qornet, you are an assistant to a group of humans.

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
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    botUser = bot.user
    name = 'UnknownBot'
    if botUser is not None:
        name = botUser.name
    print(f'Logged in as {name}')

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
async def on_message(message: discord.Message):
    if message.content.startswith('!'):
        # Process command instead of going into the executor chain
        try:
            await bot.process_commands(message)
        except Exception as e:
            # log the error trace.
            traceback.print_exception(type(e), e, e.__traceback__)
            print("oops, something went wrong when processing the bot commands");
        return
    
    guild = message.guild
    if guild is None:
        await message.channel.send("You need to be in a discord server to use the join functionality")
        return
    
    guild_id = guild.id
    log.info("currently in ", guild.name, "with member: ", guild.approximate_member_count)

    config = getConfigFromGuildId(guild_id)
    if config is None:
        # TODO: this might be invoked from DM. how to handle this?
        await message.channel.send("Invalid configuration. You need to configure me using /setup command before using this command")
        return

    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
    from langchain_community.utilities.wikidata import WikidataAPIWrapper
    from langchain_community.tools.wikidata.tool import WikidataQueryRun
    from langchain.tools import tool

    wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper(), description=(
        "A wrapper around Wikidata. Do not use when the information is in your memory"
        "Useful for when you need to answer general questions about unknown"
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be the exact name of the item you want information about "
        "or a Wikidata QID."
    ))

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    tools = [multiply]

    # Choose the LLM that will drive the agent
    llm = ChatOpenAI(model="gpt-4-1106-preview", api_key=config["OPENAI_API_KEY"])

    chatPromptTemplate = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["entities", "history"], template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE)),
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ])

    # Construct the OpenAI Functions agent
    agent = create_openai_functions_agent(llm, tools, prompt=chatPromptTemplate)

    llm = OpenAI(temperature=0, api_key=config["OPENAI_API_KEY"])
    memory = ConversationEntityMemory(llm=llm, chat_history_key="history")
    memory.entity_store = entity_store

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

    if message.author.name != "Nonuts":
      global userMessageCounter
      userMessageCounter = userMessageCounter - 1;
    
    if (bot.user and bot.user.mentioned_in(message)) or userMessageCounter < 1: 
        userMessageCounter = 100;
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

from discord.ext.voice_recv import extras
from discord.ext import commands, voice_recv
from elevenlabs.client import ElevenLabs
from io import BytesIO
from openai import OpenAI as OAI
import speech_recognition as sr
import logging

log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

def getConfigFromGuildId(guild_id: int):
    config = redis.get(str(guild_id))
    if config is None:
        return None
    return json.loads(config)


@bot.command(name="dc", description="Disconnect the bot from the voice channel it is in")
async def disconnect(ctx: commands.Context):
    author = ctx.author
    # if author is an instance of discord.User
    if isinstance(author, discord.User):
        await ctx.send("You need to be in a discord server to use the disconnect functionality")
        return

    if author.voice is None:
        await ctx.send("You need to be in a voice channel to use this command.")
        return
    
    guild = ctx.guild
    if guild is None:
        await ctx.send("You need to be in a discord server to use the disconnect functionality")
        return

    voice_client = guild.voice_client
    if voice_client is None:
        await ctx.send("The bot is not currently connected to a voice channel.")
        return

    await voice_client.disconnect(force=True)
    await ctx.send("Bot disconnected from the voice channel.")

@bot.command(name="join", description="Join the voice channel you are in")
async def join(ctx: commands.Context):
    guild = ctx.guild
    if guild is None:
        await ctx.send("You need to be in a discord server to use the join functionality")
        return
    
    guild_id = guild.id
    log.info("currently in ", guild.name, "with member: ", guild.approximate_member_count)
    if guild_id is None:
        await ctx.send("You need to be in a discord server to use the join functionality")
        return
    
    config = getConfigFromGuildId(guild_id)
    if config is None:
        await ctx.send("Invalid configuration. You need to configure me using /setup command before using this command")
        return

    xiApiKey = config["XI_API_KEY"]
    xiClient = ElevenLabs(
        api_key=xiApiKey
    )
    
    oaiClient = OAI(api_key=config["OPENAI_API_KEY"])

    message_history = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
    ]
    def text_callback(user: discord.User, output: str):
        log.info(f'{user} said {output}')

        if not output.lower().startswith("hey there"):
            return

        response = oaiClient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message_history+[
                {"role": "user", "content": output}
            ], # type: ignore
        )

        log.info(f'ai response: {response.choices[0].message.content or ""}')

        # add to message history
        message_history.append({"role": "user", "content": output})
        message_history.append({"role": "assistant", "content": response.choices[0].message.content or ""})

        aiResponse = response.choices[0].message.content or ""
        
        asource = BytesIO()
        if aiResponse is not None:
            s = xiClient.generate(
                text=aiResponse,
                voice=config["XI_VOICE_ID"],
                model="eleven_turbo_v2",
                stream=True,
            )

            for chunk in s:
                # save it to file
                asource.write(chunk)

        # send sample audio to voice channel
        asource.seek(0)

        audio_source = discord.FFmpegPCMAudio(source=asource, pipe=True)
        vc.play(audio_source)

    def cb(recognizer: sr.Recognizer, audio: sr.AudioData, user: Optional[discord.User]) -> Optional[str]:
        # log.debug("Got %s, %s, %s", audio, audio.sample_rate, audio.sample_width)
        # check audio length before sending it to whisper. make sure it's more than 100ms
        duration_ms = (len(audio.frame_data) / (audio.sample_rate * audio.sample_width)) * 1000
        if duration_ms < 100:
            return None

        text: Optional[str] = None
        try:
            func = getattr(recognizer, 'recognize_whisper_api')
            text = func(audio, api_key=config['OPENAI_API_KEY'])  # type: ignore
        except sr.UnknownValueError:
            pass
            # self._debug_audio_chunk(audio)
        return text

    sink = extras.SpeechRecognitionSink(process_cb=cb, text_cb=text_callback) #type: ignore

    author = ctx.author
    # if author is an instance of discord.User
    if isinstance(author, discord.User):
        await ctx.send("You need to be in a discord server to use the disconnect functionality")
        return

    if author.voice is None:
        await ctx.send("You need to be in a voice channel to use this command.")
        return
    
    member_voice_channel = author.voice.channel
    if member_voice_channel is None:
        await ctx.send("You need to be in a voice channel to use me!")
        return

    vc = await member_voice_channel.connect(cls=voice_recv.VoiceRecvClient)
    vc.listen(sink)

# *******************************************************************
# SETUP AND VERIFY COMMANDS
# *******************************************************************

# connect to upstash redis
from upstash_redis import Redis
import json

redis = Redis(url="https://usw1-perfect-phoenix-34606.upstash.io", token=os.getenv("UPSTASH_REDIS_GUILD_CONFIG_TOKEN", "invalid-upstash-redis-token"))

@bot.command()
async def verify_setup(ctx: commands.Context):
    await ctx.send("Verifying setup")
    
    guild = ctx.guild
    if guild is None:
        await ctx.send("You need to be in a discord server to use the join functionality")
        return
    
    guild_id = guild.id
    log.info(f"currently in {guild.name} (#{guild_id}), with total member: {guild.approximate_member_count}")
    if guild_id is None:
        await ctx.send("You need to be in a discord server to use the join functionality")
        return
    
    config = redis.get(str(guild_id))
    if config is None:
        await ctx.send("You need to set it up using /setup command")
        return

    # parse config string as json dict
    config = json.loads(config)
    
    if config.get("XI_API_KEY") is None:
        await ctx.send("You need to set elevenLabs API Key. Set it up using /setup command")
        return

    if config.get("OPENAI_API_KEY") is None:
        await ctx.send("You need to set openai key, set it up using /setup command")
        return

    await ctx.send("Your account is setup correctly!")

@bot.command()
async def setup(ctx: commands.Context):
    await ctx.send("Setting up")

@bot.command()
async def hello(ctx):
    await ctx.send(f"Hello, {ctx.author.mention}!")

bot.run(BOT_TOKEN)
