"""
Sable Bot — AI assistant for AIC Holdings' Sable portfolio management system.

Uses slack-bot-core for Slack plumbing. Sable-specific logic:
- OpenRouter API calls (chat_fn)
- SQL translation for data queries
- Sable/QA API connectors
"""
import os
import logging
import json
from typing import Optional, Dict, Any, List

import httpx
from slack_bot_core import SlackBotRunner, SlackBotConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
SABLE_API_URL = os.environ.get("SABLE_API_URL", "https://sable.aicholdings.com")
SABLE_API_KEY = os.environ.get("SABLE_API_KEY")
STATUS_CHANNEL = os.environ.get("SABLE_BOT_STATUS_CHANNEL", "C0AC77178LU")
BOT_VERSION = "2.0.0"

# OpenRouter configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-sonnet-4"

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


class OpenRouterClient:
    """Client for OpenRouter API"""

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

            payload = {"model": MODEL, "messages": messages}
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

DATA_KEYWORDS = ["how many", "show me", "list", "count", "total", "average", "sum", "find", "get", "what is", "which"]


def sable_chat(messages: List[Dict], system_prompt: Optional[str] = None) -> str:
    """
    Sable's chat_fn for SlackBotRunner.

    Handles two paths:
    1. Data queries → SQL translation → execute → summarize
    2. Everything else → straight to OpenRouter
    """
    if not openrouter:
        return "Error: OpenRouter API key not configured."

    # Get the latest user message for SQL detection
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg["content"]
            break

    # Check if this needs SQL translation
    if sable_connector and any(kw in user_message.lower() for kw in DATA_KEYWORDS):
        schema = sable_connector.get_schema()
        translation_prompt = _sql_translation_prompt(schema, user_message)
        sql_query = openrouter.chat([{"role": "user", "content": translation_prompt}])

        if not sql_query.startswith("UNABLE_TO_TRANSLATE"):
            result = sable_connector.execute_query(sql_query)
            if "error" not in result:
                summary_prompt = (
                    f'The user asked: "{user_message}"\n\n'
                    f'SQL Query executed: {sql_query}\n\n'
                    f'Results: {json.dumps(result, indent=2)}\n\n'
                    f'Provide a clear, concise summary of these results.'
                )
                return openrouter.chat(
                    [{"role": "user", "content": summary_prompt}],
                    system_prompt
                )
            else:
                return f"Error executing query: {result['error']}"

    # General conversation
    return openrouter.chat(messages, system_prompt)


def _sql_translation_prompt(schema: Dict[str, Any], user_query: str) -> str:
    """Build the SQL translation prompt"""
    schema_str = json.dumps(schema, indent=2) if schema else "Schema not available"
    return f"""You are a SQL expert. Convert the following natural language query to SQL.

Database Schema:
{schema_str}

User Query: {user_query}

Respond with ONLY the SQL query, no explanations. If you cannot create a valid SQL query, respond with "UNABLE_TO_TRANSLATE" followed by the reason."""


if __name__ == "__main__":
    config = SlackBotConfig(
        bot_name="Sable Bot",
        version=BOT_VERSION,
        system_prompt=SYSTEM_PROMPT,
        status_channel=STATUS_CHANNEL,
    )

    runner = SlackBotRunner(chat_fn=sable_chat, config=config)
    runner.start()
