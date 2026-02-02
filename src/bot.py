import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# Import the RAG pipeline. 
# Note: ensure src/pipeline.py exists and has a RAGService class with a query method.
from pipeline import RAGService
from ingestion import run_ingestion

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
# Configure global logging to track the bot's lifecycle and potential errors.
# We set the level to INFO to capture important events without flooding the logs.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Create a logger specific to this module.
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
# Load environment variables from the .env file. 
# This is crucial for keeping sensitive data like API keys secure.
load_dotenv()

# Instantiate the RAG pipeline.
# This object will be responsible for handling the logic of retrieving answers
# from the knowledge base using the Datapizza AI framework mechanics.
try:
    rag_pipeline = RAGService()
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

# Retrieve the Telegram Bot Token from environment variables.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.critical("TELEGRAM_TOKEN is missing in environment variables!")




# ------------------------------------------------------------------------------
# Command Handlers
# ------------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /start command.
    
    This function is triggered when a user sends `/start` to the bot.
    It welcomes the user and provides a brief introduction to the bot's purpose.
    """
    user_first_name = update.effective_user.first_name
    welcome_message = (
        f"Ciao {user_first_name}! 👋\n\n"
        "Sono RudyAIBot, il tuo assistente virtuale per l'Oratorio.\n"
        "Posso rispondere alle tue domande riguardanti regolamenti, orari e attività, "
        "basandomi sui documenti ufficiali.\n\n"
        "Prova a chiedermi qualcosa! Esempio:\n"
        "🗣 'Quali sono gli orari di apertura?'\n"
        "🗣 'Come funzionano le iscrizioni?'"
    )
    # Send the welcome message back to the chat where the command originated.
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /help command.
    
    This function is triggered when a user sends `/help`.
    It explains how to interact with the bot.
    """
    help_text = (
        "🤖 **Guida Rapida**\n\n"
        "Basta scrivermi una domanda in linguaggio naturale e cercherò la risposta "
        "nei documenti ufficiali dell'Oratorio.\n\n"
        "**Comandi disponibili:**\n"
        "/start - Avvia il bot\n"
        "/help - Mostra questo messaggio\n"
        "/update_kb - (Admin) Aggiorna la conoscenza del bot"
    )
    # Parse mode Markdown allows for bold text formatting in the Telegram message.
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=help_text, 
        parse_mode='Markdown'
    )

async def update_kb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /update_kb command.
    
    This function is triggered when a user sends `/update_kb`.
    It updates the knowledge base of the bot.
    """
    try:
        # Run the ingestion script
        await run_ingestion()
        
        # Send a success message
        await update.message.reply_text("Knowledge base updated successfully!")
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")
        await update.message.reply_text("Error updating knowledge base.")




# ------------------------------------------------------------------------------
# Message Handlers
# ------------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for text messages that are NOT commands.
    
    This function processes the user's question, sends it to the RAG pipeline,
    and returns the generated answer.
    """
    # 1. Extract the text of the message from the user.
    user_query = update.message.text
    chat_type = update.effective_chat.type

    # Logica per i gruppi: rispondi solo se menzionato
    if chat_type in ["group", "supergroup"]:
        if not (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id) and f"@{context.bot.username}" not in user_query:
            return # Ignora il messaggio se non è una menzione o una risposta al bot
    
    # 2. Indicate typing status to show the bot is "thinking".
    # This improves UX by giving immediate feedback.
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        if rag_pipeline:
            # 3. Query the RAG Pipeline.
            # This is where the heavy lifting happens: semantic search + LLM generation.
            response = rag_pipeline.query(user_query)
        else:
            response = "⚠️ Sistema RAG non inizializzato. Impossibile rispondere."

        # 4. specific reply to the user's message.
        # Check if the response exceeds Telegram's max message length (4096 characters)
        max_length = 4096
        if len(response) > max_length:
            # Split the message into chunks
            for i in range(0, len(response), max_length):
                chunk = response[i:i + max_length]
                await update.message.reply_text(chunk)
        else:
            await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error processing message '{user_query}': {e}")
        error_message = "😓 Scusa, si è verificato un errore nel processare la tua richiesta."
        await update.message.reply_text(error_message)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Global Error Handler.
    
    Logs any unhandled exceptions that occur during the execution of other handlers.
    """
    logger.error(msg="Exception while handling an update:", exc_info=context.error)




# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    """
    Main entry point for the bot.
    
    Configures the Telegram Application, registers handlers, and starts polling.
    """
    if not TELEGRAM_TOKEN:
        logger.error("Cannot start bot without a valid TELEGRAM_TOKEN. Exiting.")
        return

    # Create the Application object using the token.
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register Command Handlers
    # These respond to commands starting with '/'
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(CommandHandler('update_kb', update_kb))

    # Register Message Handler
    # This filter ensures we only handle text messages that are not commands.
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    # Register Error Handler
    application.add_error_handler(error_handler)

    # Start the Bot
    # run_polling() keeps the script running and listening for new updates from Telegram.
    logger.info("🤖 RudyAIBot is starting polling...")
    application.run_polling()

if __name__ == '__main__':
    main()
