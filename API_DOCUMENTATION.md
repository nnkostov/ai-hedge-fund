# AI Hedge Fund System - Comprehensive API Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [REST API Documentation](#rest-api-documentation)
4. [Core Components](#core-components)
5. [Agent System](#agent-system)
6. [Trading Tools](#trading-tools)
7. [LLM Integration](#llm-integration)
8. [Backtesting System](#backtesting-system)
9. [Frontend Components](#frontend-components)
10. [Configuration & Setup](#configuration--setup)
11. [Examples & Usage](#examples--usage)

---

## Overview

The AI Hedge Fund system is a comprehensive trading platform that uses multiple AI agents to make investment decisions. The system combines traditional financial analysis with modern AI capabilities to provide automated trading recommendations.

### Key Features
- Multi-agent AI analysis system
- Support for multiple LLM providers (OpenAI, Anthropic, Groq, etc.)
- Real-time trading recommendations
- Historical backtesting capabilities
- REST API for programmatic access
- Modern web interface

---

## System Architecture

```
AI Hedge Fund System
├── Backend (FastAPI)
│   ├── REST API Endpoints
│   ├── Data Models & Schemas
│   ├── Business Logic Services
│   └── Database Integration
├── Core Engine (src/)
│   ├── AI Agents
│   ├── Trading Tools
│   ├── LLM Integration
│   ├── Graph Workflow
│   └── Backtesting Engine
├── Frontend (React + TypeScript)
│   ├── Trading Dashboard
│   ├── Agent Configuration
│   └── Results Visualization
└── External Integrations
    ├── Financial Data APIs
    ├── LLM Providers
    └── Market Data Sources
```

---

## REST API Documentation

### Base URL
```
http://localhost:8000
```

### 1. Hedge Fund Operations

#### POST /hedge-fund/run
Execute trading analysis using selected AI agents.

**Request Body:**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "selected_agents": ["warren_buffett", "peter_lynch", "cathie_wood"],
  "agent_models": [
    {
      "agent_id": "warren_buffett",
      "model_name": "gpt-4o",
      "model_provider": "OpenAI"
    }
  ],
  "start_date": "2024-01-01",
  "end_date": "2024-03-31",
  "model_name": "gpt-4o",
  "model_provider": "OpenAI",
  "initial_cash": 100000.0,
  "margin_requirement": 0.0
}
```

**Response:** Server-Sent Events (SSE) stream
```json
{
  "event": "progress",
  "data": {
    "agent": "warren_buffett_agent",
    "ticker": "AAPL",
    "status": "Analyzing fundamentals",
    "timestamp": "2024-01-01T10:00:00Z"
  }
}
```

**Final Response:**
```json
{
  "event": "complete",
  "data": {
    "decisions": {
      "AAPL": {
        "signal": "bullish",
        "action": "buy",
        "quantity": 100,
        "confidence": 85.5,
        "reasoning": "Strong fundamentals and growth prospects"
      }
    },
    "analyst_signals": {
      "warren_buffett_agent": {
        "AAPL": {
          "signal": "bullish",
          "confidence": 90,
          "reasoning": "Excellent ROE and competitive moat"
        }
      }
    }
  }
}
```

#### GET /hedge-fund/agents
Get list of available AI agents.

**Response:**
```json
{
  "agents": [
    {
      "id": "warren_buffett",
      "name": "Warren Buffett",
      "description": "Value investing focused on fundamentals"
    },
    {
      "id": "peter_lynch",
      "name": "Peter Lynch", 
      "description": "Growth at reasonable price strategy"
    }
  ]
}
```

#### GET /hedge-fund/language-models
Get list of available LLM models.

**Response:**
```json
{
  "models": [
    {
      "display_name": "GPT-4o",
      "model_name": "gpt-4o",
      "provider": "OpenAI"
    },
    {
      "display_name": "Claude 3.5 Sonnet",
      "model_name": "claude-3-5-sonnet-20241022",
      "provider": "Anthropic"
    }
  ]
}
```

### 2. Health Check

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T10:00:00Z",
  "version": "0.1.0"
}
```

### 3. Storage Operations

#### GET /storage/flows
List saved trading flows.

#### POST /storage/flows
Save a new trading flow configuration.

#### GET /storage/flows/{flow_id}
Retrieve specific flow by ID.

#### PUT /storage/flows/{flow_id}
Update existing flow.

#### DELETE /storage/flows/{flow_id}
Delete flow.

---

## Core Components

### 1. Main Trading Engine

**File:** `src/main.py`

#### `run_hedge_fund()`
Main function to execute trading analysis.

```python
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI"
) -> dict
```

**Parameters:**
- `tickers`: List of stock symbols to analyze
- `start_date`: Analysis start date (YYYY-MM-DD)
- `end_date`: Analysis end date (YYYY-MM-DD)
- `portfolio`: Current portfolio state
- `show_reasoning`: Whether to display agent reasoning
- `selected_analysts`: List of agent IDs to use
- `model_name`: LLM model to use
- `model_provider`: LLM provider name

**Returns:**
```python
{
    "decisions": dict,  # Final trading decisions
    "analyst_signals": dict  # Individual agent signals
}
```

**Example Usage:**
```python
from src.main import run_hedge_fund

portfolio = {
    "cash": 100000.0,
    "margin_requirement": 0.0,
    "positions": {"AAPL": {"long": 0, "short": 0}}
}

result = run_hedge_fund(
    tickers=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-03-31",
    portfolio=portfolio,
    selected_analysts=["warren_buffett", "peter_lynch"]
)
```

### 2. Graph State Management

**File:** `src/graph/state.py`

#### `AgentState`
Core state structure for the agent workflow.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]
```

#### `show_agent_reasoning()`
Display agent analysis output.

```python
def show_agent_reasoning(output, agent_name: str) -> None
```

---

## Agent System

### Available Agents

1. **Warren Buffett Agent** (`warren_buffett.py`)
   - Focus: Value investing, fundamentals analysis
   - Key metrics: ROE, debt levels, competitive moats

2. **Peter Lynch Agent** (`peter_lynch.py`)
   - Focus: Growth at reasonable price (GARP)
   - Key metrics: PEG ratio, earnings growth

3. **Cathie Wood Agent** (`cathie_wood.py`)
   - Focus: Disruptive innovation
   - Key metrics: Technology adoption, market disruption

4. **Stanley Druckenmiller Agent** (`stanley_druckenmiller.py`)
   - Focus: Macro trends, momentum
   - Key metrics: Economic indicators, trend analysis

5. **Michael Burry Agent** (`michael_burry.py`)
   - Focus: Contrarian value investing
   - Key metrics: Market inefficiencies, deep value

6. **Bill Ackman Agent** (`bill_ackman.py`)
   - Focus: Activist investing
   - Key metrics: Corporate governance, catalysts

7. **Charlie Munger Agent** (`charlie_munger.py`)
   - Focus: Quality businesses at fair prices
   - Key metrics: Business quality, management

8. **Phil Fisher Agent** (`phil_fisher.py`)
   - Focus: Growth investing
   - Key metrics: Innovation, market leadership

9. **Ben Graham Agent** (`ben_graham.py`)
   - Focus: Classical value investing
   - Key metrics: Book value, asset valuation

10. **Aswath Damodaran Agent** (`aswath_damodaran.py`)
    - Focus: Valuation methodology
    - Key metrics: DCF models, risk analysis

11. **Rakesh Jhunjhunwala Agent** (`rakesh_jhunjhunwala.py`)
    - Focus: Indian market expertise
    - Key metrics: Emerging market dynamics

### Agent Structure

Each agent follows this standard structure:

```python
def agent_function(state: AgentState) -> dict:
    """
    Analyze stocks using agent-specific methodology.
    
    Args:
        state: Current workflow state containing data and metadata
        
    Returns:
        dict: Updated state with agent analysis
    """
    # 1. Extract data from state
    data = state["data"]
    tickers = data["tickers"]
    
    # 2. Perform analysis for each ticker
    analysis = {}
    for ticker in tickers:
        # Fetch required data
        # Apply agent-specific analysis
        # Generate signal and reasoning
        pass
    
    # 3. Return results
    return {
        "messages": [HumanMessage(content=json.dumps(analysis))],
        "data": state["data"]
    }
```

### Signal Format

All agents return signals in this standardized format:

```python
class AgentSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0-100
    reasoning: str
```

---

## Trading Tools

### 1. Financial Data API

**File:** `src/tools/api.py`

#### `get_prices()`
Fetch historical price data.

```python
def get_prices(
    ticker: str, 
    start_date: str, 
    end_date: str
) -> list[Price]
```

#### `get_financial_metrics()`
Fetch financial metrics and ratios.

```python
def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10
) -> list[FinancialMetrics]
```

#### `get_company_news()`
Fetch company news and events.

```python
def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000
) -> list[CompanyNews]
```

#### `get_insider_trades()`
Fetch insider trading data.

```python
def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000
) -> list[InsiderTrade]
```

#### `search_line_items()`
Search for specific financial statement line items.

```python
def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10
) -> list[LineItem]
```

### 2. Data Models

**File:** `src/data/models.py`

Key data structures for financial data:

```python
class Price(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    ticker: str

class FinancialMetrics(BaseModel):
    ticker: str
    period_ending_date: str
    period: str
    return_on_equity: float | None
    debt_to_equity: float | None
    operating_margin: float | None
    current_ratio: float | None
    # ... many more fields

class CompanyNews(BaseModel):
    ticker: str
    date: str
    headline: str
    summary: str
    url: str
```

---

## LLM Integration

### 1. Model Configuration

**File:** `src/llm/models.py`

#### Supported Providers

```python
class ModelProvider(str, Enum):
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
```

#### `get_model()`
Create LLM instance for specific provider.

```python
def get_model(
    model_name: str, 
    model_provider: ModelProvider
) -> ChatOpenAI | ChatGroq | ChatOllama | None
```

**Example Usage:**
```python
from src.llm.models import get_model, ModelProvider

# OpenAI GPT-4
llm = get_model("gpt-4o", ModelProvider.OPENAI)

# Anthropic Claude
llm = get_model("claude-3-5-sonnet-20241022", ModelProvider.ANTHROPIC)

# Local Ollama
llm = get_model("llama2", ModelProvider.OLLAMA)
```

### 2. Model Features

#### `LLMModel.has_json_mode()`
Check if model supports structured JSON output.

#### `LLMModel.is_custom()`
Check if model requires custom configuration.

---

## Backtesting System

### 1. Backtester Class

**File:** `src/backtester.py`

#### Initialization

```python
class Backtester:
    def __init__(
        self,
        agent: Callable,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        model_name: str = "gpt-4o",
        model_provider: str = "OpenAI",
        selected_analysts: list[str] = [],
        initial_margin_requirement: float = 0.0
    )
```

#### `run_backtest()`
Execute historical trading simulation.

```python
def run_backtest(self) -> dict
```

**Features:**
- Long and short position support
- Margin trading simulation
- Performance metrics calculation
- Real-time progress display

#### `execute_trade()`
Process individual trade execution.

```python
def execute_trade(
    self,
    ticker: str,
    action: str,
    quantity: float,
    current_price: float
) -> int
```

**Supported Actions:**
- `buy`: Purchase long position
- `sell`: Close long position
- `short`: Open short position
- `cover`: Close short position

### 2. Performance Metrics

The backtester calculates comprehensive performance metrics:

```python
{
    "sharpe_ratio": float,
    "sortino_ratio": float,
    "max_drawdown": float,
    "total_return": float,
    "volatility": float,
    "long_short_ratio": float,
    "gross_exposure": float,
    "net_exposure": float
}
```

---

## Frontend Components

### Technology Stack
- **React 18** with TypeScript
- **Vite** for build tooling
- **Tailwind CSS** for styling
- **Shadcn/ui** for components

### Key Components

1. **Trading Dashboard**
   - Real-time agent analysis display
   - Portfolio performance visualization
   - Trade execution interface

2. **Agent Configuration**
   - Select active agents
   - Configure LLM models per agent
   - Set analysis parameters

3. **Results Visualization**
   - Interactive charts
   - Performance metrics
   - Historical analysis

---

## Configuration & Setup

### 1. Environment Variables

Create a `.env` file with required API keys:

```bash
# Financial Data
FINANCIAL_DATASETS_API_KEY=your_key_here

# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
GOOGLE_API_KEY=your_google_key

# Ollama (for local models)
OLLAMA_HOST=localhost
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Installation

```bash
# Install Python dependencies
poetry install

# Install frontend dependencies
cd app/frontend
npm install
```

### 3. Running the System

```bash
# Start backend
cd app/backend
python -m uvicorn main:app --reload

# Start frontend
cd app/frontend
npm run dev

# Run CLI version
python -m src.main --tickers AAPL,MSFT --start-date 2024-01-01
```

---

## Examples & Usage

### 1. Basic Trading Analysis

```python
from src.main import run_hedge_fund

# Configure portfolio
portfolio = {
    "cash": 100000.0,
    "margin_requirement": 0.0,
    "positions": {
        "AAPL": {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0},
        "MSFT": {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}
    },
    "realized_gains": {
        "AAPL": {"long": 0.0, "short": 0.0},
        "MSFT": {"long": 0.0, "short": 0.0}
    }
}

# Run analysis
result = run_hedge_fund(
    tickers=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-03-31",
    portfolio=portfolio,
    selected_analysts=["warren_buffett", "peter_lynch", "cathie_wood"],
    model_name="gpt-4o",
    model_provider="OpenAI"
)

# Process results
for ticker, decision in result["decisions"].items():
    print(f"{ticker}: {decision['action']} {decision['quantity']} shares")
    print(f"Confidence: {decision['confidence']:.1f}%")
    print(f"Reasoning: {decision['reasoning']}")
```

### 2. Backtesting Example

```python
from src.backtester import Backtester
from src.main import run_hedge_fund

# Create backtester
backtester = Backtester(
    agent=run_hedge_fund,
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100000.0,
    selected_analysts=["warren_buffett", "peter_lynch"]
)

# Run backtest
performance = backtester.run_backtest()

print(f"Total Return: {performance['total_return']:.2f}%")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2f}%")
```

### 3. Custom Agent Implementation

```python
from src.graph.state import AgentState
from langchain_core.messages import HumanMessage
import json

def custom_agent(state: AgentState) -> dict:
    """Custom trading agent implementation."""
    data = state["data"]
    tickers = data["tickers"]
    
    analysis = {}
    for ticker in tickers:
        # Your custom analysis logic here
        analysis[ticker] = {
            "signal": "bullish",
            "confidence": 75.0,
            "reasoning": "Custom analysis indicates strong buy signal"
        }
    
    # Update analyst signals
    state["data"]["analyst_signals"]["custom_agent"] = analysis
    
    return {
        "messages": [HumanMessage(content=json.dumps(analysis))],
        "data": state["data"]
    }
```

### 4. API Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Run hedge fund analysis
response = requests.post(f"{BASE_URL}/hedge-fund/run", json={
    "tickers": ["AAPL", "MSFT"],
    "selected_agents": ["warren_buffett", "peter_lynch"],
    "start_date": "2024-01-01",
    "end_date": "2024-03-31",
    "model_name": "gpt-4o",
    "model_provider": "OpenAI"
})

# Process streaming response
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').split('data: ')[1])
        if data['event'] == 'complete':
            decisions = data['data']['decisions']
            print("Trading Decisions:", decisions)
```

---

## Error Handling

### Common Error Codes

- **400**: Invalid request parameters
- **401**: Authentication failed (missing API keys)
- **429**: Rate limit exceeded
- **500**: Internal server error

### Rate Limiting

The system implements intelligent rate limiting for external API calls:
- Automatic retry with exponential backoff
- Configurable retry attempts
- Graceful degradation on persistent failures

### Data Validation

All API inputs are validated using Pydantic models:
- Type checking
- Range validation
- Required field enforcement
- Custom validation rules

---

## Performance Considerations

### Caching Strategy

The system implements multi-level caching:
1. **In-memory cache** for frequently accessed data
2. **File-based cache** for persistent storage
3. **API response caching** to minimize external calls

### Optimization Tips

1. **Batch API calls** when analyzing multiple tickers
2. **Use date ranges** efficiently to minimize data fetching
3. **Select relevant agents** based on investment strategy
4. **Configure appropriate timeframes** for analysis

---

## Security

### API Key Management
- Store sensitive keys in environment variables
- Use separate keys for different environments
- Implement key rotation policies

### Data Protection
- No sensitive data stored in logs
- Secure transmission of all API calls
- Input sanitization and validation

---

## Monitoring & Logging

### Health Checks
- `/health` endpoint for system status
- Database connectivity checks
- External API availability monitoring

### Performance Metrics
- Response time tracking
- Error rate monitoring
- Resource utilization metrics

---

This documentation provides comprehensive coverage of the AI Hedge Fund system's public APIs, components, and usage patterns. For additional support or questions, refer to the inline code documentation or system logs.