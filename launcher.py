import asyncio
import os

from dotenv import load_dotenv
load_dotenv(override=True)

from bot import Norobonut

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", 'invalid_token')

async def run_bot():
    async with Norobonut() as bot:
        await bot.start(BOT_TOKEN)

def main():
    """Launches the bot."""
    asyncio.run(run_bot())

if __name__ == '__main__':
    main()
