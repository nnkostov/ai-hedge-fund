# AI Hedge Fund System - Quick Start Guide

## Table of Contents
1. [Installation & Setup](#installation--setup)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Common Use Cases](#common-use-cases)
4. [Troubleshooting](#troubleshooting)
5. [Next Steps](#next-steps)

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd ai-hedge-fund

# Install Python dependencies
poetry install

# Install frontend dependencies
cd app/frontend
npm install
cd ../..
```

### 2. Environment Configuration

Create `.env` file in project root:

```bash
# Required: Financial Data API
FINANCIAL_DATASETS_API_KEY=your_financial_api_key

# Required: At least one LLM provider
OPENAI_API_KEY=your_openai_key
# OR
ANTHROPIC_API_KEY=your_anthropic_key
# OR
GROQ_API_KEY=your_groq_key

# Optional: Additional LLM providers
DEEPSEEK_API_KEY=your_deepseek_key
GOOGLE_API_KEY=your_google_key

# Optional: Local Ollama setup
OLLAMA_HOST=localhost
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Quick Test

Test your setup with a simple analysis:

```bash
python -m src.main \
  --tickers AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --initial-cash 100000
```

---

## Basic Usage Examples

### 1. CLI Analysis

#### Single Stock Analysis
```bash
python -m src.main \
  --tickers AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --show-reasoning
```

#### Multi-Stock Portfolio
```bash
python -m src.main \
  --tickers AAPL,MSFT,GOOGL \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --initial-cash 500000 \
  --show-reasoning
```

#### Custom Date Range
```bash
python -m src.main \
  --tickers TSLA \
  --start-date 2023-06-01 \
  --end-date 2023-12-31 \
  --initial-cash 100000
```

### 2. Python Script Usage

#### Basic Analysis Script

```python
# basic_analysis.py
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
    selected_analysts=["warren_buffett", "peter_lynch"],
    model_name="gpt-4o",
    model_provider="OpenAI"
)

# Print results
for ticker, decision in result["decisions"].items():
    print(f"\n{ticker} Analysis:")
    print(f"Action: {decision['action']}")
    print(f"Quantity: {decision['quantity']}")
    print(f"Confidence: {decision['confidence']:.1f}%")
    print(f"Reasoning: {decision['reasoning']}")
```

#### Automated Daily Analysis

```python
# daily_analysis.py
from datetime import datetime, timedelta
from src.main import run_hedge_fund
import json

def daily_analysis(tickers, lookback_days=30):
    """Run daily analysis for given tickers."""
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Initialize portfolio
    portfolio = {
        "cash": 100000.0,
        "margin_requirement": 0.0,
        "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0} for ticker in tickers},
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
    }
    
    # Run analysis
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        selected_analysts=["warren_buffett", "peter_lynch", "cathie_wood"]
    )
    
    # Save results
    with open(f"analysis_{end_date}.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result

# Usage
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    result = daily_analysis(tickers)
    print("Daily analysis complete. Results saved.")
```

### 3. API Usage

#### Start Backend Server

```bash
cd app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### API Client Example

```python
# api_client.py
import requests
import json
import time

class HedgeFundAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def run_analysis(self, tickers, selected_agents, **kwargs):
        """Run hedge fund analysis via API."""
        
        payload = {
            "tickers": tickers,
            "selected_agents": selected_agents,
            "start_date": kwargs.get("start_date", "2024-01-01"),
            "end_date": kwargs.get("end_date", "2024-03-31"),
            "model_name": kwargs.get("model_name", "gpt-4o"),
            "model_provider": kwargs.get("model_provider", "OpenAI"),
            "initial_cash": kwargs.get("initial_cash", 100000.0)
        }
        
        response = requests.post(
            f"{self.base_url}/hedge-fund/run",
            json=payload,
            stream=True
        )
        
        # Process streaming response
        final_result = None
        for line in response.iter_lines():
            if line:
                try:
                    # Parse SSE data
                    data_str = line.decode('utf-8')
                    if data_str.startswith('data: '):
                        data = json.loads(data_str[6:])  # Remove 'data: ' prefix
                        
                        if data.get('event') == 'progress':
                            print(f"Progress: {data['data']['agent']} - {data['data']['status']}")
                        elif data.get('event') == 'complete':
                            final_result = data['data']
                            break
                except:
                    continue
        
        return final_result
    
    def get_agents(self):
        """Get available agents."""
        response = requests.get(f"{self.base_url}/hedge-fund/agents")
        return response.json()
    
    def get_models(self):
        """Get available LLM models."""
        response = requests.get(f"{self.base_url}/hedge-fund/language-models")
        return response.json()

# Usage
if __name__ == "__main__":
    api = HedgeFundAPI()
    
    # Get available resources
    agents = api.get_agents()
    models = api.get_models()
    
    print("Available agents:", [agent['name'] for agent in agents['agents']])
    print("Available models:", [model['display_name'] for model in models['models']])
    
    # Run analysis
    result = api.run_analysis(
        tickers=["AAPL", "MSFT"],
        selected_agents=["warren_buffett", "peter_lynch"],
        start_date="2024-01-01",
        end_date="2024-03-31"
    )
    
    print("\nAnalysis Results:")
    for ticker, decision in result['decisions'].items():
        print(f"{ticker}: {decision['action']} {decision['quantity']} shares")
```

### 4. Web Interface Usage

#### Start Full Stack

```bash
# Terminal 1: Start backend
cd app/backend
python -m uvicorn main:app --reload

# Terminal 2: Start frontend
cd app/frontend
npm run dev
```

Access web interface at: `http://localhost:5173`

---

## Common Use Cases

### 1. Daily Stock Screening

```python
# stock_screener.py
from src.main import run_hedge_fund
from datetime import datetime, timedelta

def screen_stocks(watchlist, min_confidence=70):
    """Screen stocks and filter by confidence threshold."""
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    strong_buys = []
    strong_sells = []
    
    for ticker in watchlist:
        # Analyze individual stock
        portfolio = {
            "cash": 100000.0,
            "margin_requirement": 0.0,
            "positions": {ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}},
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0}}
        }
        
        result = run_hedge_fund(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            selected_analysts=["warren_buffett", "peter_lynch", "cathie_wood"]
        )
        
        decision = result["decisions"][ticker]
        
        if decision["confidence"] >= min_confidence:
            if decision["action"].lower() in ["buy", "cover"]:
                strong_buys.append((ticker, decision))
            elif decision["action"].lower() in ["sell", "short"]:
                strong_sells.append((ticker, decision))
    
    return strong_buys, strong_sells

# Usage
watchlist = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"]
buys, sells = screen_stocks(watchlist, min_confidence=75)

print("Strong Buy Signals:")
for ticker, decision in buys:
    print(f"{ticker}: {decision['confidence']:.1f}% confidence")

print("\nStrong Sell Signals:")
for ticker, decision in sells:
    print(f"{ticker}: {decision['confidence']:.1f}% confidence")
```

### 2. Portfolio Rebalancing

```python
# portfolio_rebalancer.py
from src.main import run_hedge_fund

def rebalance_portfolio(current_positions, target_allocation, total_value):
    """Suggest portfolio rebalancing based on AI analysis."""
    
    tickers = list(current_positions.keys())
    
    # Get current analysis
    result = run_hedge_fund(
        tickers=tickers,
        start_date="2024-01-01",
        end_date="2024-03-31",
        portfolio={
            "cash": 0.0,  # Assume fully invested
            "margin_requirement": 0.0,
            "positions": current_positions,
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers}
        },
        selected_analysts=["warren_buffett", "peter_lynch"]
    )
    
    # Calculate recommended allocation adjustments
    recommendations = {}
    
    for ticker in tickers:
        decision = result["decisions"][ticker]
        current_allocation = current_positions[ticker]["long"] * get_current_price(ticker) / total_value
        target_weight = target_allocation.get(ticker, 0.0)
        
        # Adjust target based on AI signal
        if decision["signal"] == "bullish" and decision["confidence"] > 75:
            adjusted_target = min(target_weight * 1.2, 0.25)  # Max 25% position
        elif decision["signal"] == "bearish" and decision["confidence"] > 75:
            adjusted_target = max(target_weight * 0.8, 0.02)  # Min 2% position
        else:
            adjusted_target = target_weight
        
        recommendations[ticker] = {
            "current": current_allocation,
            "target": adjusted_target,
            "adjustment": adjusted_target - current_allocation,
            "confidence": decision["confidence"]
        }
    
    return recommendations

# Example usage
current_positions = {
    "AAPL": {"long": 100, "short": 0},
    "MSFT": {"long": 80, "short": 0},
    "GOOGL": {"long": 50, "short": 0}
}

target_allocation = {
    "AAPL": 0.30,
    "MSFT": 0.30,
    "GOOGL": 0.40
}

recommendations = rebalance_portfolio(current_positions, target_allocation, 500000)
```

### 3. Backtesting Strategy

```python
# backtest_strategy.py
from src.backtester import Backtester
from src.main import run_hedge_fund

def backtest_multi_timeframe(tickers, periods):
    """Backtest strategy across multiple timeframes."""
    
    results = {}
    
    for period_name, (start_date, end_date) in periods.items():
        print(f"Backtesting {period_name}: {start_date} to {end_date}")
        
        backtester = Backtester(
            agent=run_hedge_fund,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000.0,
            selected_analysts=["warren_buffett", "peter_lynch", "cathie_wood"]
        )
        
        performance = backtester.run_backtest()
        results[period_name] = performance
    
    return results

# Define test periods
periods = {
    "2023_bull": ("2023-01-01", "2023-06-30"),
    "2023_correction": ("2023-07-01", "2023-12-31"),
    "2024_ytd": ("2024-01-01", "2024-03-31")
}

# Run backtests
tickers = ["AAPL", "MSFT", "GOOGL"]
results = backtest_multi_timeframe(tickers, periods)

# Compare results
for period, performance in results.items():
    print(f"\n{period}:")
    print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A'):.2f}")
    print(f"  Max Drawdown: {performance.get('max_drawdown', 'N/A'):.2f}%")
```

### 4. Risk Management Integration

```python
# risk_manager.py
from src.main import run_hedge_fund

class RiskManager:
    def __init__(self, max_position_size=0.25, max_sector_allocation=0.40):
        self.max_position_size = max_position_size
        self.max_sector_allocation = max_sector_allocation
    
    def apply_risk_limits(self, decisions, portfolio_value):
        """Apply risk management rules to trading decisions."""
        
        adjusted_decisions = {}
        
        for ticker, decision in decisions.items():
            original_quantity = decision["quantity"]
            
            # Calculate position size as % of portfolio
            estimated_value = original_quantity * self.get_current_price(ticker)
            position_size = estimated_value / portfolio_value
            
            # Apply position size limit
            if position_size > self.max_position_size:
                scale_factor = self.max_position_size / position_size
                adjusted_quantity = int(original_quantity * scale_factor)
                
                adjusted_decisions[ticker] = {
                    **decision,
                    "quantity": adjusted_quantity,
                    "risk_adjustment": f"Position size reduced by {(1-scale_factor)*100:.1f}%"
                }
            else:
                adjusted_decisions[ticker] = decision
        
        return adjusted_decisions
    
    def get_current_price(self, ticker):
        # Implementation to get current price
        return 150.0  # Placeholder

# Usage
risk_manager = RiskManager()

# Get AI recommendations
result = run_hedge_fund(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-03-31",
    portfolio={"cash": 100000.0},
    selected_analysts=["warren_buffett"]
)

# Apply risk management
risk_adjusted = risk_manager.apply_risk_limits(
    result["decisions"], 
    portfolio_value=100000.0
)
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Errors
**Error:** `API key not found` or `Authentication failed`

**Solution:**
```bash
# Check .env file exists and has correct keys
cat .env | grep API_KEY

# Verify key format (no quotes, no extra spaces)
OPENAI_API_KEY=sk-your-key-here
```

#### 2. Import Errors
**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Run from project root directory
cd /path/to/ai-hedge-fund

# Ensure Python path includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use module syntax
python -m src.main --tickers AAPL
```

#### 3. Data Fetch Errors
**Error:** `Error fetching data: ticker - 429 - Rate limited`

**Solution:**
- Wait for rate limit reset (usually 60 seconds)
- Use smaller ticker lists
- Implement retry logic in custom scripts

#### 4. LLM Model Errors
**Error:** `Model not found` or `Invalid model name`

**Solution:**
```python
# Check available models
from src.llm.models import LLM_ORDER
print("Available models:", [model[1] for model in LLM_ORDER])

# Use correct model name
model_name = "gpt-4o"  # Correct
model_name = "gpt4"    # Incorrect
```

### Performance Tips

1. **Cache Data:** The system automatically caches API responses. For development, data persists between runs.

2. **Batch Analysis:** Analyze multiple tickers in one call rather than individual calls.

3. **Select Relevant Agents:** Use fewer agents for faster analysis when appropriate.

4. **Optimize Date Ranges:** Shorter date ranges reduce data fetching time.

---

## Next Steps

### Advanced Usage

1. **Custom Agent Development**
   - Study existing agents in `src/agents/`
   - Implement custom analysis logic
   - Integrate with LLM reasoning

2. **Integration Projects**
   - Connect to trading platforms
   - Build custom dashboards
   - Implement automated trading

3. **Model Experimentation**
   - Test different LLM providers
   - Compare agent combinations
   - Optimize for specific strategies

### Resources

- **Full API Documentation:** `API_DOCUMENTATION.md`
- **Component Reference:** `COMPONENT_REFERENCE.md` 
- **System Architecture:** Review `src/` directory structure
- **Example Scripts:** Create custom scripts based on patterns above

### Community & Support

- Review existing issues and discussions
- Contribute improvements and bug fixes
- Share custom agents and strategies

---

This quick start guide provides practical examples for immediately using the AI Hedge Fund system. Start with basic CLI usage, then progress to API integration and custom development based on your needs.