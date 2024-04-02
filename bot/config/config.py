from typing import Optional, Any
import os

# This class is the configuration for the bot.
class ServerBotConfig:
  # OpenAI API key
  openai_api_key: str

  # Eleven labs API key
  xi_api_key: str

  # Eleven labs voice ID
  xi_voice_id: str | None

  # which channels to listen to?
  on_message_channel_ids: list[str] | None

  def __init__(self, openai_api_key: str, xi_api_key: str, xi_voice_id: Optional[str] = None, on_message_channel_ids: Optional[list[str]] = None):
    self.openai_api_key = openai_api_key
    self.xi_api_key = xi_api_key
    self.xi_voice_id = xi_voice_id
    self.on_message_channel_ids = on_message_channel_ids

  # initializer for JSON object or map
  @staticmethod
  def from_json(jsonData: dict[str, Any]):
    openai_api_key = jsonData.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if openai_api_key is None: 
      raise ValueError("Invalid JSON: OPENAI_API_KEY is required")
    
    xi_api_key = jsonData.get("XI_API_KEY") or os.getenv("XI_API_KEY")
    if xi_api_key is None:
      raise ValueError("Invalid JSON: XI_API_KEY is required")

    return ServerBotConfig(
      openai_api_key=openai_api_key,
      xi_api_key=xi_api_key,
      xi_voice_id=jsonData.get("XI_VOICE_ID") or os.getenv("XI_VOICE_ID"),
      on_message_channel_ids=jsonData.get("on_message_channel_ids")
    )
  
  # convert to dictionary
  def to_dict(self):
    return {
      "OPENAI_API_KEY": self.openai_api_key,
      "XI_API_KEY": self.xi_api_key,
      "XI_VOICE_ID": self.xi_voice_id,
      "on_message_channel_ids": self.on_message_channel_ids
    }

