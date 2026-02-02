# bot-sable

Slack bot for Sable portfolio data queries using semantic-to-SQL translation.

## Overview

bot-sable enables natural language queries against Sable portfolio data:

- **Portfolio queries** - "How many open positions do we have?"
- **P&L tracking** - "What's our YTD P&L?"
- **Position data** - "Show me our top 10 positions by value"

The bot uses AI to translate natural language to SQL queries, executes them against Sable's API, and returns summarized results.

## Quick Start

### Prerequisites

- Python 3.11+
- Slack App with Socket Mode enabled
- OpenRouter API key (to be migrated to Artemis)
- Sable API access

### Installation

```bash
git clone https://github.com/aic-holdings/bot-sable.git
cd bot-sable
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Running Locally

```bash
python app.py
```

### Deployment

Deploy to Railway:

```bash
railway link
railway up
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SLACK_BOT_TOKEN` | Slack bot token (xoxb-...) | Yes |
| `SLACK_APP_TOKEN` | Slack app token (xapp-...) | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `SABLE_API_URL` | Sable API base URL | Yes |
| `SABLE_API_KEY` | Sable API authentication | Yes |
| `SABLE_BOT_STATUS_CHANNEL` | Channel for status messages | No |

## Future Work

- [ ] Migrate to Artemis AI (centralized tracking)
- [ ] Refactor to modular structure (src/ layout)
- [ ] Add FastAPI health endpoint

## Related Projects

- [bot-slack-core](https://github.com/aic-holdings/bot-slack-core) - Shared Slack bot utilities
- [Sable](https://sable.aicholdings.com) - Portfolio management system

## License

Proprietary - AIC Holdings
