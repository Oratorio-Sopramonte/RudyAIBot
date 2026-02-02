import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, User, Chat, Message
from src.bot import handle_message

@pytest.mark.asyncio
async def test_end_to_end_query(mocker):
    """
    Simulate a user sending a message to the bot and verify the response.
    This tests the integration of handle_message with the RAG pipeline.
    """
    # 1. Mock the RAG Pipeline in bot.py
    # Since bot.py instantiates RAGService at module level, we need to patch the instance
    # that is already created or patch the class before import if possible.
    # A cleaner way given the structure is to patch 'src.bot.rag_pipeline'
    
    # Create a mock pipeline instance
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = "The Oratorio is open from 8 AM to 8 PM."
    
    # Patch the global variable in bot.py
    mocker.patch("src.bot.rag_pipeline", mock_pipeline)
    
    # 2. Construct a Mock Telegram Update
    mock_update = MagicMock(spec=Update)
    mock_message = MagicMock(spec=Message)
    mock_user = MagicMock(spec=User)
    mock_chat = MagicMock(spec=Chat)
    
    mock_user.first_name = "TestUser"
    mock_chat.id = 12345
    
    mock_message.text = "What are the opening hours?"
    mock_message.reply_text = AsyncMock() # Must be async
    
    mock_update.message = mock_message
    mock_update.effective_user = mock_user
    mock_update.effective_chat = mock_chat
    
    # Mock context
    mock_context = MagicMock()
    mock_context.bot.send_chat_action = AsyncMock()
    
    # 3. Call handle_message
    await handle_message(mock_update, mock_context)
    
    # 4. Verify RAG was queried with user text
    mock_pipeline.query.assert_called_once_with("What are the opening hours?")
    
    # 5. Verify Bot replied with the answer
    mock_message.reply_text.assert_called_once_with("The Oratorio is open from 8 AM to 8 PM.")
