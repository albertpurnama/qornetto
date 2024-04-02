CREATE TABLE discord_ai_user_conversation (
    id SERIAL PRIMARY KEY,
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    conversation_start_date TIMESTAMP NOT NULL,
    messages JSONB NOT NULL,
    CONSTRAINT max_messages_per_row CHECK (jsonb_array_length(messages) <= 100)
);

