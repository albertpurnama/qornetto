import logging
import discord
from discord import app_commands
from discord.ext import commands
import traceback

from dotenv import load_dotenv
load_dotenv(override=True)

import os
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", 'invalid_token')

from typing import  Optional

from openai.types.chat import ChatCompletionMessageParam,ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

MY_GUILD=discord.Object(id=1162069309766508564)  # replace with your guild id

from db import SessionLocal
from models import DiscordAIUserConversation

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        # A CommandTree is a special type that holds all the application command
        # state required to make it work. This is a separate class because it
        # allows all the extra state to be opt-in.
        # Whenever you want to work with application commands, your tree is used
        # to store and work with them.
        # Note: When using commands.Bot instead of discord.Client, the bot will
        # maintain its own tree instead.
        self.tree = app_commands.CommandTree(self)

    # In this basic example, we just synchronize the app commands to one guild.
    # Instead of specifying a guild to every command, we copy over our global commands instead.
    # By doing so, we don't have to wait up to an hour until they are shown to the end-user.
    # async def setup_hook(self):
    #     # This copies the global commands over to your guild.
    #     self.tree.copy_global_to(guild=MY_GUILD)
    #     syncedCommands = await self.tree.sync(guild=MY_GUILD)
    #     print(f"Synced {len(syncedCommands)} commands")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
client = MyClient(intents=intents)

@client.tree.command(name="sync", description="Sync the bot's commands to the server")
@app_commands.guild_only()
async def sync(interaction: discord.Interaction):
    await interaction.response.send_message("Syncing commands...")
    await client.tree.sync(guild=discord.Object(id=interaction.guild_id)) # type: ignore
    await interaction.response.send_message("Commands synced.")

@client.event
async def on_ready():
    user = client.user
    if user is None:
        print("User is None")
        return
    print(f'Logged in as {user} (ID: {user.id})')
    print('------')

import psycopg2

@client.event
async def on_message(message: discord.Message):
    # TODO: maybe use debug tool Jishaku instead of
    # triggering sync through message
    if message.content == "!sync_nonuts_global_commands":
        await client.tree.sync()
        return

    if message.author.bot:
        return
    
    if message.guild is None:
        # we're ignoring DMs for now.
        return

    config = getConfigFromGuildId(message.guild.id)
    if config is None:
        return

    if str(message.channel.id) not in (config.on_message_channel_ids or []):
        # ignore messages not coming from the user.
        return
    
    session = SessionLocal()
    try:
        # Query the existing conversation
        conversation = session.query(DiscordAIUserConversation).filter_by(
            guild_id=message.guild.id, channel_id=message.channel.id).first()

        if conversation:
            # Parse the existing messages, append the new one, and trim if necessary
            messages = conversation.messages
            messages_list = json.loads(json.dumps(messages))
            messages_list.append({"author": message.author.name, "bot": False, "content": message.content})
            total_chars = sum(len(m["content"]) for m in messages_list)
            while len(messages_list) > 100 or total_chars > 100000:
                removed_message = messages.pop(0)
                total_chars -= len(removed_message["content"])
            conversation.messages = messages_list
            messages = messages_list
        else:
            # Create a new conversation record
            conversation = DiscordAIUserConversation(
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                conversation_start_date=message.created_at.isoformat(),
                messages=[{"author": message.author.name, "bot": False, "content": message.content}]
            )
            messages = conversation.messages
            session.add(conversation)

        # Commit the changes
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    
    # now that we have the messages, let's create OpenAI 
    # conversation messages from it.
    # We need to make sure that the type of messages is
    # appropriate for OpenAI

    openai_conversation_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant living in a discord channel. Answer concisely."),
    ]

    for m in messages:
        if m.get('bot', False) == True:
            openai_conversation_messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=m.get("content", "")))
        else:
            openai_conversation_messages.append(ChatCompletionUserMessageParam(role="user", content=f'{m.get("author", "someone")}: {m.get("content", "someone")}'))

    async with message.channel.typing():
        # pass in the messages to gpt-3.5 openai
        # Generate the OpenAI conversation
        try:
            openaiClient = OAI(api_key=config.openai_api_key)
            response = openaiClient.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=openai_conversation_messages,
            )
            generated_response = response.choices[0].message.content

            # Send the generated response back to the discord channel
            await message.channel.send(generated_response)
            
            # save message to database
            messages.append({
                "author": "assistant",
                "bot": True,
                "content": generated_response,
            })

            session.query(DiscordAIUserConversation).filter(
                DiscordAIUserConversation.guild_id == message.guild.id,
                DiscordAIUserConversation.channel_id == message.channel.id
            ).update({"messages": messages})
            session.commit()

        except Exception as e:
            log.error(f"Failed to generate OpenAI conversation: {e}")
            await message.channel.send("Sorry, I couldn't generate a response. Please try again later.")

    # cleanup
    session.close()

from discord.ext.voice_recv import extras
from discord.ext import commands, voice_recv
from elevenlabs.client import ElevenLabs
from io import BytesIO
from openai import OpenAI as OAI
import speech_recognition as sr
import logging
from models.config import ServerBotConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def getConfigFromGuildId(guild_id: int) -> ServerBotConfig | None:
    config = redis.get(str(guild_id))
    if config is None:
        return ServerBotConfig.from_json({}) # return default config (using hosted OpenAI API key)
    return ServerBotConfig.from_json(json.loads(config))

@client.tree.command(name="dc", description="Disconnect the bot from the voice channel it is in")
async def disconnect(ctx: discord.Interaction):
    guild = ctx.guild
    if guild is None:
        await ctx.response.send_message("You need to be in a discord server to use the disconnect functionality")
        return

    voice_client = guild.voice_client
    if voice_client is None:
        await ctx.response.send_message("The bot is not currently connected to a voice channel.")
        return

    await voice_client.disconnect(force=True)
    await ctx.response.send_message("Bot disconnected from the voice channel.")

@client.tree.command(name="join", description="Join the voice channel you are in")
async def join(ctx: discord.Interaction):
    guild_id = ctx.guild_id
    log.info(f"currently in guild_id: {guild_id}")
    if guild_id is None:
        await ctx.response.send_message("You need to be in a discord server to use the join functionality")
        return
    
    config = getConfigFromGuildId(guild_id)
    if config is None:
        await ctx.response.send_message("Invalid configuration. You need to configure me using /setup command before using this command")
        return

    xiApiKey = config.xi_api_key
    xiClient = ElevenLabs(
        api_key=xiApiKey
    )
    
    oaiClient = OAI(api_key=config.openai_api_key)

    def text_callback(user: discord.User, output: str):
        log.info(f'{user} said {output}')

        if not output.lower().startswith("hey there"):
            return
    
        conn = psycopg2.connect(os.getenv('DATABASE_DSN'))
        
        # rolling window of up to 100 last messages until max context of 10000 characters.
        cursor = conn.cursor()
        cursor.execute("SELECT messages FROM discord_ai_user_conversation WHERE guild_id = %s AND channel_id = %s", (ctx.guild_id, ctx.channel_id))
        rows = cursor.fetchall()
        # read the json messages
        if rows:
            messages = rows[0][0]
            # Add the new message to the list of messages
            messages.append({
                "author": user.name,
                "bot": False,
                "content": output,
            })

            # Ensure we don't exceed 100 messages or 100k characters total
            total_chars = sum(len(m["content"]) for m in messages)
            while len(messages) > 100 or total_chars > 100000:
                removed_message = messages.pop(0)
                total_chars -= len(removed_message["content"])

            # Update the database with the new list of messages
            cursor.execute("UPDATE discord_ai_user_conversation SET messages = %s WHERE guild_id = %s AND channel_id = %s", 
                        (json.dumps(messages), ctx.guild_id, ctx.channel_id))
            
            conn.commit()
        else:
            # create a new list of messages
            messages = [{
                "author": user.name,
                "bot": False,
                "content": output
            }]

            from datetime import datetime

            cursor.execute("INSERT INTO discord_ai_user_conversation (guild_id, channel_id, conversation_start_date, messages) VALUES (%s, %s, %s, %s)", 
                        (ctx.guild_id, ctx.channel_id, datetime.now().isoformat(), json.dumps(messages)))
            conn.commit()

        message_history: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant in a discord channel. Answer concisely."},
        ]

        for m in messages:
            if m["bot"]:
                message_history.append(ChatCompletionAssistantMessageParam(role="assistant", content=m.get("content", "")))
            else:
                message_history.append(ChatCompletionUserMessageParam(role="user", content=f'{m["author"]}: {m["content"]}'))

        response = oaiClient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message_history+[
                {"role": "user", "content": output}
            ],
        )

        log.info(f'ai response: {response.choices[0].message.content or ""}')

        aiResponse = response.choices[0].message.content or ""

        messages.append({
            "author": "assistant",
            "bot": True,
            "content": aiResponse,
        })

        # save message to db
        cursor.execute("UPDATE discord_ai_user_conversation SET messages = %s WHERE guild_id = %s AND channel_id = %s", 
                        (json.dumps(messages), ctx.guild_id, ctx.channel_id))
        conn.commit()
        conn.close()

        asource = BytesIO()
        if aiResponse is not None:
            currentChannel = ctx.channel
            if config is None or not isinstance(currentChannel, discord.VoiceChannel):
                currentChannel.send("Configuration not found") # type: ignore
                return
            s = xiClient.generate(
                text=aiResponse,
                voice=(config.xi_voice_id or 'None'), # type: ignore
                model="eleven_turbo_v2",
                stream=True,
            )

            for chunk in s:
                # save it to file
                if isinstance(chunk, bytes):
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
            text = func(audio, api_key=config.openai_api_key)
        except sr.UnknownValueError:
            pass
            # self._debug_audio_chunk(audio)
        return text

    sink = extras.SpeechRecognitionSink(process_cb=cb, text_cb=text_callback) #type: ignore

    # check if the current channel is voice channel
    currentVoiceChannel = ctx.channel
    if not isinstance(currentVoiceChannel, discord.VoiceChannel):
        await ctx.response.send_message("This command can only be used in a voice channel.")
        return

    vc = await currentVoiceChannel.connect(cls=voice_recv.VoiceRecvClient)
    vc.listen(sink)
    await ctx.response.send_message("Listening in this channel now")

# SETUP AND VERIFY COMMANDS
# *******************************************************************

# connect to upstash redis
from upstash_redis import Redis
import json
from views.key_setup import KeySetup

redis = Redis(
    url=os.getenv("UPSTASH_REDIS_GUILD_CONFIG_URL", "invalid-redis-url"), 
    token=os.getenv("UPSTASH_REDIS_GUILD_CONFIG_TOKEN", "invalid-upstash-redis-token"),
)

@client.tree.command()
async def verify_setup(ctx: discord.Interaction):
    guild_id = ctx.guild_id
    log.info(f"currently in guild#{guild_id})")
    if guild_id is None:
        await ctx.response.send_message("You need to be in a discord server to use the join functionality")
        return
    
    config = getConfigFromGuildId(guild_id)
    if config is None:
        await ctx.response.send_message("Bot is not configured. You need to set it up using /setup command")
        return

    await ctx.response.send_message("Your account is setup correctly!")

@client.tree.command()
async def setup(ctx: discord.Interaction):
    await ctx.response.send_modal(KeySetup(redis=redis))

@client.tree.command()
@commands.guild_only()
async def listen(ctx: discord.Interaction):
    print(f"listen called by {ctx.user} in {ctx.guild_id}")
    config = getConfigFromGuildId(ctx.guild_id or 0)
    if config is None:
        await ctx.response.send_message("Invalid configuration. You need to configure me using /setup command before using this command")
        return
    
    # append current channel to on_message_channel_ids
    config.on_message_channel_ids = config.on_message_channel_ids or []
    config.on_message_channel_ids.append(str(ctx.channel_id))

    # save to redis
    redis.set(str(ctx.guild_id), json.dumps(config.to_dict()))

    await ctx.response.send_message("Listening in this channel now")

@client.tree.command()
async def hello(ctx):
    await ctx.send(f"Hello, {ctx.author.mention}!")

client.run(BOT_TOKEN)
