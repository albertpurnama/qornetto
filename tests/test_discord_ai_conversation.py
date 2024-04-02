from bot.models import DiscordAIUserConversation
import json

def test_addition():
    assert 1 + 1 == 2

def test_discord_user_conversation_json():
    conv = DiscordAIUserConversation(guild_id=123, channel_id=456, messages=[])
    assert json.dumps(conv.messages) == json.dumps([])

