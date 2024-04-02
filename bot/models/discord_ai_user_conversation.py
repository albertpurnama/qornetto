from sqlalchemy import Column, BIGINT, Integer, Index
from sqlalchemy.schema import CheckConstraint
from sqlalchemy.dialects import postgresql
from .base import Base

class DiscordAIUserConversation(Base):
    __tablename__ = "discord_ai_user_conversation"

    id = Column(Integer, primary_key=True)
    guild_id = Column(BIGINT, nullable=False)
    channel_id = Column(BIGINT, nullable=False)
    conversation_start_date = Column(postgresql.TIMESTAMP, nullable=False)
    messages = Column(postgresql.JSONB, nullable=False)
    
    __table_args__ = (
        Index('ix_guild_id_channel_id', 'guild_id', 'channel_id'),
        CheckConstraint('jsonb_array_length(messages) <= 100', name='max_messages_per_row'),
    )
