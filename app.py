import os
import logging
import re
import signal
import sys
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import httpx
import json
from typing import Optional, Dict, Any, List

# Configure logging - v1.2.1
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SABLE_API_URL = os.environ.get("SABLE_API_URL", "https://sable.aicholdings.com")
SABLE_QA_API_URL = os.environ.get("SABLE_QA_API_URL", "https://qa.sable.jettaintelligence.com")
SABLE_API_KEY = os.environ.get("SABLE_API_KEY")
STATUS_CHANNEL = os.environ.get("SABLE_BOT_STATUS_CHANNEL", "C0AC77178LU")  # #sable-bot
BOT_VERSION = "1.3.0"

# Track startup time for uptime calculation
import time
_start_time = 0.0

# Initialize Slack app
app = App(token=SLACK_BOT_TOKEN)

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"  # Claude Sonnet 4


class SableDataConnector:
    """Connector for Sable data APIs"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.Client(timeout=30.0)
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Execute a SQL query against Sable database"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = self.client.post(
                f"{self.api_url}/api/query",
                headers=headers,
                json={"query": sql}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {"error": str(e)}
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema for semantic translation"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.client.get(
                f"{self.api_url}/api/schema",
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Schema fetch error: {e}")
            return {"error": str(e)}


class SemanticToSQL:
    """Translates natural language to SQL using Claude"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def get_translation_prompt(self, user_query: str) -> str:
        schema_str = json.dumps(self.schema, indent=2) if self.schema else "Schema not available"
        return f"""You are a SQL expert. Convert the following natural language query to SQL.

Database Schema:
{schema_str}

User Query: {user_query}

Respond with ONLY the SQL query, no explanations. If you cannot create a valid SQL query, respond with "UNABLE_TO_TRANSLATE" followed by the reason."""


class OpenRouterClient:
    """Client for OpenRouter API with Claude Sonnet 4.5"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(timeout=60.0)
    
    def chat(self, messages: List[Dict], system_prompt: Optional[str] = None) -> str:
        """Send a chat completion request to OpenRouter"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://sable-bot.railway.app",
                "X-Title": "Sable Bot"
            }
            
            payload = {
                "model": MODEL,
                "messages": messages
            }
            
            if system_prompt:
                payload["messages"] = [{"role": "system", "content": system_prompt}] + messages
            
            response = self.client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            return f"Error communicating with AI: {str(e)}"


# Initialize clients
sable_connector = SableDataConnector(SABLE_API_URL, SABLE_API_KEY) if SABLE_API_KEY else None
openrouter = OpenRouterClient(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else None

# System prompt for Sable Bot
SYSTEM_PROMPT = """You are Sable Bot, a helpful AI assistant for AIC Holdings.

ABOUT SABLE (your main expertise):
Sable is a portfolio management and P&L tracking system for hedge funds.
- Web app: https://sable.aicholdings.com
- Features: Position tracking, trade history, daily P&L, risk analytics

CONVERSATION STYLE:
- Be friendly and conversational - you can chat about anything
- Remember what users say in the conversation and refer back to it
- If someone tells you they like hot dogs, remember that!
- For Sable questions: give specific, actionable answers
- For access questions: direct to dshanklin@aicholdings.com
- Don't constantly redirect every message back to Sable - be natural

You're a helpful assistant who happens to specialize in Sable, but you can have normal conversations too."""


# Cache for bot user ID (fetched once at startup)
_bot_user_id: Optional[str] = None


def get_thread_history(channel: str, thread_ts: str, limit: int = 20) -> List[Dict]:
    """Fetch thread history from Slack to provide conversation context"""
    logger.info(f"Fetching thread history: channel={channel}, thread_ts={thread_ts}")
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                "https://slack.com/api/conversations.replies",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                params={"channel": channel, "ts": thread_ts, "limit": limit}
            )
            data = response.json()
            if data.get("ok"):
                messages = data.get("messages", [])
                logger.info(f"Got {len(messages)} messages from thread")
                return messages
            else:
                logger.warning(f"Slack API error: {data.get('error')}")
    except httpx.TimeoutException:
        logger.error("Timeout fetching thread history")
    except Exception as e:
        logger.error(f"Error fetching thread history: {e}")
    return []


def get_diagnostic_info() -> str:
    """Generate diagnostic information about the bot"""
    global _start_time

    uptime_seconds = int(time.time() - _start_time) if _start_time else 0
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    # Check service status
    openrouter_status = "Connected" if openrouter else "Not configured"
    sable_status = "Connected" if sable_connector else "Not configured"

    return f"""*Sable Bot Diagnostics*

:robot_face: *Version:* {BOT_VERSION}
:clock1: *Uptime:* {uptime_str}
:brain: *AI Backend:* OpenRouter ({openrouter_status})
:chart_with_upwards_trend: *Sable API:* {sable_status}
:gear: *Model:* {MODEL}
:house: *Platform:* Railway

*Environment:*
• Sable URL: {SABLE_API_URL}
• Status Channel: {STATUS_CHANNEL}
"""


def build_conversation_messages(thread_messages: List[Dict]) -> List[Dict]:
    """Convert Slack thread messages to OpenRouter conversation format"""
    messages = []
    for msg in thread_messages:
        text = msg.get("text", "")
        # Remove bot mentions from text
        text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        if not text:
            continue

        # Determine if this is from the bot or user
        is_bot = msg.get("bot_id") is not None
        role = "assistant" if is_bot else "user"
        messages.append({"role": role, "content": text})

    logger.info(f"Built conversation with {len(messages)} messages: {[m['role'] for m in messages]}")
    return messages


def process_sable_query(user_message: str, conversation_history: List[Dict] = None) -> str:
    """Process a user query about Sable data with conversation context"""
    if not openrouter:
        return "Error: OpenRouter API key not configured."

    # Build messages with conversation history
    if conversation_history:
        messages = conversation_history
    else:
        messages = [{"role": "user", "content": user_message}]

    # Check if this needs SQL translation
    if sable_connector and any(keyword in user_message.lower() for keyword in
        ["how many", "show me", "list", "count", "total", "average", "sum", "find", "get", "what is", "which"]):

        # Get schema for translation
        schema = sable_connector.get_schema()
        translator = SemanticToSQL(schema)

        # Translate to SQL
        translation_prompt = translator.get_translation_prompt(user_message)
        sql_query = openrouter.chat([{"role": "user", "content": translation_prompt}])

        if not sql_query.startswith("UNABLE_TO_TRANSLATE"):
            # Execute the query
            result = sable_connector.execute_query(sql_query)

            if "error" not in result:
                # Have Claude summarize the results
                summary_prompt = f"""The user asked: "{user_message}"

SQL Query executed: {sql_query}

Results: {json.dumps(result, indent=2)}

Provide a clear, concise summary of these results."""
                return openrouter.chat([{"role": "user", "content": summary_prompt}], SYSTEM_PROMPT)
            else:
                return f"Error executing query: {result['error']}"

    # For general questions, use Claude with full conversation history
    return openrouter.chat(messages, SYSTEM_PROMPT)


@app.event("app_mention")
def handle_mention(event, say, client):
    """Handle @mentions of the bot - always reply in thread with conversation context"""
    global _bot_user_id

    user_message = event.get("text", "")
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")  # Use existing thread or start new one

    # Remove the bot mention from the message
    user_message = re.sub(r"<@[A-Z0-9]+>", "", user_message).strip()

    if not user_message:
        say("Hi! I'm Sable Bot. Ask me anything!", thread_ts=thread_ts)
        return

    # Check for diagnostic commands
    lower_msg = user_message.lower()
    if lower_msg in ["status", "info", "diag", "diagnostics", "version", "health", "ping"]:
        say(get_diagnostic_info(), thread_ts=thread_ts)
        return

    try:
        # Cache bot user ID on first use
        if _bot_user_id is None:
            auth_info = client.auth_test()
            _bot_user_id = auth_info.get("user_id", "")

        # Get thread history for conversation context
        conversation_history = None
        logger.info(f"Processing message: '{user_message[:50]}...' in thread={thread_ts}")
        if thread_ts:
            thread_messages = get_thread_history(channel, thread_ts)
            if thread_messages:
                conversation_history = build_conversation_messages(thread_messages)
                logger.info(f"Using conversation history with {len(conversation_history)} messages")
            else:
                logger.info("No thread messages found, using single message")
        else:
            logger.info("No thread_ts, using single message")

        response = process_sable_query(user_message, conversation_history)
        say(response, thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        say(f"Sorry, I encountered an error: {str(e)}", thread_ts=thread_ts)


@app.event("message")
def handle_message(event, say):
    """Handle direct messages to the bot"""
    # Only respond to DMs (channel_type: im)
    if event.get("channel_type") != "im":
        return
    
    # Ignore bot messages
    if event.get("bot_id"):
        return
    
    user_message = event.get("text", "")
    
    try:
        response = process_sable_query(user_message)
        say(response)
    except Exception as e:
        logger.error(f"Error processing DM: {e}")
        say(f"Sorry, I encountered an error: {str(e)}")


def post_status_message(message: str) -> None:
    """Post a status message to the status channel"""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={"channel": STATUS_CHANNEL, "text": message}
            )
            data = response.json()
            if not data.get("ok"):
                logger.error(f"Failed to post status: {data.get('error')}")
    except Exception as e:
        logger.error(f"Error posting status message: {e}")


def shutdown_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutdown signal received...")
    post_status_message(f":warning: Sable Bot v{BOT_VERSION} is shutting down for updates...")
    sys.exit(0)


if __name__ == "__main__":
    if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
        logger.error("Missing Slack tokens. Set SLACK_BOT_TOKEN and SLACK_APP_TOKEN.")
        exit(1)

    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. AI features will be disabled.")

    # Register shutdown handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Track startup time and post startup message
    _start_time = time.time()
    logger.info(f"Starting Sable Bot v{BOT_VERSION}...")
    post_status_message(f":white_check_mark: Sable Bot v{BOT_VERSION} is online! (Railway restart complete)")

    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
