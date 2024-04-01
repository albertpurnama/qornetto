import discord
import json
import traceback
from upstash_redis import Redis

class KeySetup(discord.ui.Modal, title='Nonut Setup'):
    def __init__(self, redis: Redis):
        self.redis = redis

    # Our modal classes MUST subclass `discord.ui.Modal`,
    # but the title can be whatever you want.

    # This will be a short input, where the user can enter their name
    # It will also have a placeholder, as denoted by the `placeholder` kwarg.
    # By default, it is required and is a short-style input which is exactly
    # what we want.
    openai_api_key = discord.ui.TextInput(
        label='OpenAI API Key',
        placeholder='sk-193123...',
    )

    xi_api_key = discord.ui.TextInput(
        label='ElevenLabs API Key',
        placeholder='d9dkj2344...',
    )

    async def on_submit(self, interaction: discord.Interaction):
        # save to redis
        self.redis.set(str(interaction.guild_id), json.dumps({
            "OPENAI_API_KEY": self.openai_api_key.value,
            "XI_API_KEY": self.xi_api_key.value,
        }))
        await interaction.response.send_message(f'I have been configured. You can verify it by running /verify_setup', ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Something went wrong.', ephemeral=True)

        # Make sure we know what the error actually is
        traceback.print_exception(type(error), error, error.__traceback__)
