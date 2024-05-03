from __future__ import annotations


from discord.ext import commands
import discord
from typing import TYPE_CHECKING, List, Optional
import logging

initial_extensions = [
    "cogs.polling"
]

testing_guild_id = 1162069309766508564

description = """
Hello! I am a bot written by Albert to provide some nice utilities.
"""

def _prefix_callable(bot: Norobonut, msg: discord.Message):
    user_id = bot.user.id
    base = [f'<@!{user_id}> ', f'<@{user_id}> ']
    if msg.guild is None:
        base.append('!')
        base.append('?')
    else:
        # base.extend(bot.prefixes.get(msg.guild.id, ['?', '!']))
        pass
    return base

class Norobonut(commands.Bot):
    user: discord.ClientUser

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        allowed_mentions = discord.AllowedMentions(roles=False, everyone=False, users=True)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        super().__init__(
            command_prefix=_prefix_callable,
            description=description,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True,
        )

    async def setup_hook(self) -> None:

        # here, we are loading extensions prior to sync to ensure we are syncing interactions defined in those extensions.

        for extension in initial_extensions:
            try:
                await self.load_extension(extension)
            except Exception as e:
                logging.exception('Failed to load extension %s.', extension)


        # In overriding setup hook,
        # we can do things that require a bot prior to starting to process events from the websocket.
        # In this case, we are using this to ensure that once we are connected, we sync for the testing guild.
        # You should not do this for every guild or for global sync, those should only be synced when changes happen.
        if testing_guild_id:
            guild = discord.Object(testing_guild_id)
            # We'll copy in the global commands to test with:
            self.tree.copy_global_to(guild=guild)
            # followed by syncing to the testing guild.
            await self.tree.sync(guild=guild)

        # This would also be a good place to connect to our database and
        # load anything that should be in memory prior to handling events.

