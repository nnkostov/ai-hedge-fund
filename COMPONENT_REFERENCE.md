# AI Hedge Fund System - Component Reference Guide

## Table of Contents
1. [Agent Components](#agent-components)
2. [Data Models](#data-models)
3. [Utility Functions](#utility-functions)
4. [Backend Services](#backend-services)
5. [Graph Workflow](#graph-workflow)
6. [Progress Tracking](#progress-tracking)
7. [Display & Visualization](#display--visualization)

---

## Agent Components

### Agent Base Structure

All agents in the system follow a standardized interface and implementation pattern:

#### Core Agent Interface

```python
def agent_function(state: AgentState) -> dict:
    """
    Standard agent interface for trading analysis.
    
    Args:
        state (AgentState): Current workflow state containing:
            - messages: Message history
            - data: Trading data (tickers, dates, portfolio)
            - metadata: Configuration (model, reasoning flags)
    
    Returns:
        dict: Updated state with agent analysis
    """
```

#### Agent Implementation Pattern

```python
# 1. Data Extraction
data = state["data"]
tickers = data["tickers"]
end_date = data["end_date"]
start_date = data["start_date"]

# 2. Progress Tracking
progress.update_status("agent_name", ticker, "Starting analysis")

# 3. Data Fetching
metrics = get_financial_metrics(ticker, end_date)
prices = get_prices(ticker, start_date, end_date)
news = get_company_news(ticker, end_date, start_date)

# 4. Analysis Logic
analysis_result = perform_analysis(metrics, prices, news)

# 5. LLM Integration (if needed)
llm_output = generate_llm_output(analysis_result, state)

# 6. State Update
state["data"]["analyst_signals"]["agent_name"] = analysis_result

# 7. Return Results
return {
    "messages": [HumanMessage(content=json.dumps(analysis_result))],
    "data": state["data"]
}
```

### Warren Buffett Agent Implementation

**File:** `src/agents/warren_buffett.py`

#### Key Analysis Functions

##### `analyze_fundamentals(metrics: list) -> dict`
Evaluates company fundamentals using Buffett's criteria:
- **ROE Analysis**: Targets >15% return on equity
- **Debt Assessment**: Prefers debt-to-equity <0.5
- **Margin Evaluation**: Looks for operating margins >15%
- **Liquidity Check**: Current ratio >1.5

```python
def analyze_fundamentals(metrics: list) -> dict[str, any]:
    latest_metrics = metrics[0]
    score = 0
    reasoning = []
    
    # ROE Check (2 points max)
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    
    # Debt Check (2 points max)
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    
    return {"score": score, "details": "; ".join(reasoning)}
```

##### `analyze_moat(metrics: list) -> dict`
Comprehensive competitive advantage analysis:
- **ROE Consistency**: 80%+ periods with ROE >15%
- **Margin Stability**: Stable/improving operating margins
- **Asset Efficiency**: Revenue per dollar of assets
- **Performance Stability**: Low coefficient of variation

```python
def analyze_moat(metrics: list) -> dict[str, any]:
    moat_score = 0
    max_score = 5
    reasoning = []
    
    # ROE Consistency Analysis
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity]
    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        
        if roe_consistency >= 0.8:
            moat_score += 2
            reasoning.append("Excellent ROE consistency indicates durable advantage")
    
    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning)
    }
```

##### `calculate_owner_earnings(financial_line_items: list) -> dict`
Buffett's preferred earnings measure:
```
Owner Earnings = Net Income + Depreciation - Maintenance CapEx - Working Capital Changes
```

### Agent Signal Generation

#### LLM-Enhanced Analysis

```python
def generate_buffett_output(ticker: str, analysis_data: dict, state: AgentState) -> WarrenBuffettSignal:
    """Generate final Buffett analysis using LLM reasoning."""
    
    # Prepare analysis context
    context = {
        "ticker": ticker,
        "fundamental_score": analysis_data["fundamental_analysis"]["score"],
        "moat_analysis": analysis_data["moat_analysis"],
        "margin_of_safety": analysis_data["margin_of_safety"]
    }
    
    # Create LLM prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", buffett_system_prompt),
        ("user", json.dumps(context))
    ])
    
    # Generate analysis
    llm_output = call_llm(prompt, state, WarrenBuffettSignal)
    
    return llm_output
```

---

## Data Models

### Core Financial Data Models

**File:** `src/data/models.py`

#### Price Data Model

```python
class Price(BaseModel):
    time: str                 # ISO timestamp
    open: float              # Opening price
    high: float              # Daily high
    low: float               # Daily low
    close: float             # Closing price
    volume: int              # Trading volume
    ticker: str              # Stock symbol
    
    def to_ohlc(self) -> dict:
        """Convert to OHLC format for charting."""
        return {
            'timestamp': self.time,
            'o': self.open,
            'h': self.high,
            'l': self.low,
            'c': self.close,
            'v': self.volume
        }
```

#### Financial Metrics Model

```python
class FinancialMetrics(BaseModel):
    ticker: str
    period_ending_date: str
    period: str                    # "ttm", "quarterly", "annual"
    
    # Profitability Metrics
    return_on_equity: float | None
    return_on_assets: float | None
    return_on_invested_capital: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    
    # Valuation Metrics
    price_to_earnings: float | None
    price_to_book: float | None
    price_to_sales: float | None
    enterprise_value_to_ebitda: float | None
    
    # Financial Health
    current_ratio: float | None
    quick_ratio: float | None
    debt_to_equity: float | None
    debt_to_assets: float | None
    
    # Growth Metrics
    revenue_growth: float | None
    earnings_growth: float | None
    book_value_growth: float | None
    
    # Market Data
    market_cap: float | None
    enterprise_value: float | None
    shares_outstanding: int | None
```

#### Company News Model

```python
class CompanyNews(BaseModel):
    ticker: str
    date: str                     # ISO date
    headline: str
    summary: str
    url: str
    sentiment: str | None = None  # "positive", "negative", "neutral"
    relevance_score: float | None = None  # 0-1 relevance to ticker
    
    def is_recent(self, days: int = 7) -> bool:
        """Check if news is within specified days."""
        news_date = datetime.fromisoformat(self.date.split('T')[0])
        cutoff = datetime.now() - timedelta(days=days)
        return news_date >= cutoff
```

---

## Utility Functions

### Display Utilities

**File:** `src/utils/display.py`

#### `print_trading_output(result: dict) -> None`
Main function for displaying trading results with color-coded formatting.

```python
def print_trading_output(result: dict) -> None:
    """Format and display comprehensive trading results."""
    
    decisions = result.get("decisions")
    analyst_signals = result.get("analyst_signals", {})
    
    for ticker, decision in decisions.items():
        # Print ticker header
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        
        # Format analyst signals table
        table_data = []
        for agent, signals in analyst_signals.items():
            if ticker in signals and agent != "risk_management_agent":
                signal = signals[ticker]
                table_data.append([
                    format_agent_name(agent),
                    format_signal(signal.get("signal")),
                    format_confidence(signal.get("confidence")),
                    format_reasoning(signal.get("reasoning"))
                ])
        
        # Display formatted table
        print(tabulate(table_data, headers=["Agent", "Signal", "Confidence", "Reasoning"]))
```

#### Color Formatting Functions

```python
def format_signal(signal: str) -> str:
    """Apply color coding to trading signals."""
    signal_colors = {
        "BULLISH": Fore.GREEN,
        "BEARISH": Fore.RED,
        "NEUTRAL": Fore.YELLOW
    }
    color = signal_colors.get(signal.upper(), Fore.WHITE)
    return f"{color}{signal.upper()}{Style.RESET_ALL}"

def format_confidence(confidence: float) -> str:
    """Format confidence percentage with appropriate color."""
    if confidence >= 80:
        color = Fore.GREEN
    elif confidence >= 60:
        color = Fore.YELLOW
    else:
        color = Fore.RED
    return f"{color}{confidence}%{Style.RESET_ALL}"
```

### Progress Tracking

**File:** `src/utils/progress.py`

#### Progress Manager

```python
class ProgressManager:
    def __init__(self):
        self.handlers = []
        self.is_running = False
    
    def register_handler(self, handler: Callable):
        """Register progress update handler."""
        self.handlers.append(handler)
    
    def update_status(self, agent: str, ticker: str | None, status: str, analysis: str = None):
        """Update progress status and notify handlers."""
        timestamp = datetime.now().isoformat()
        
        for handler in self.handlers:
            try:
                handler(agent, ticker, status, analysis, timestamp)
            except Exception as e:
                print(f"Progress handler error: {e}")
    
    def start(self):
        """Start progress tracking."""
        self.is_running = True
    
    def stop(self):
        """Stop progress tracking."""
        self.is_running = False

# Global progress instance
progress = ProgressManager()
```

---

## Backend Services

### Graph Service

**File:** `app/backend/services/graph.py`

#### `create_graph(selected_agents: list[str]) -> StateGraph`
Dynamically create agent workflow graph.

```python
def create_graph(selected_agents: list[str]) -> StateGraph:
    """Create workflow graph with selected agents."""
    
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start_node)
    
    # Add analyst nodes
    analyst_nodes = get_analyst_nodes()
    for agent_key in selected_agents:
        if agent_key in analyst_nodes:
            node_name, node_func = analyst_nodes[agent_key]
            workflow.add_node(node_name, node_func)
            workflow.add_edge("start_node", node_name)
    
    # Add management nodes
    workflow.add_node("risk_management", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)
    
    # Connect workflow
    for agent_key in selected_agents:
        if agent_key in analyst_nodes:
            node_name = analyst_nodes[agent_key][0]
            workflow.add_edge(node_name, "risk_management")
    
    workflow.add_edge("risk_management", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)
    
    workflow.set_entry_point("start_node")
    return workflow
```

#### `run_graph_async(graph, portfolio, tickers, ...)`
Execute graph workflow asynchronously.

```python
async def run_graph_async(
    graph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request: HedgeFundRequest
) -> dict:
    """Execute trading workflow asynchronously."""
    
    initial_state = {
        "messages": [HumanMessage(content="Execute trading analysis")],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {}
        },
        "metadata": {
            "model_name": model_name,
            "model_provider": model_provider,
            "show_reasoning": False
        }
    }
    
    # Execute graph
    result = await graph.ainvoke(initial_state)
    return result
```

### Portfolio Service

**File:** `app/backend/services/portfolio.py`

#### `create_portfolio(initial_cash, margin_requirement, tickers)`
Initialize portfolio structure.

```python
def create_portfolio(
    initial_cash: float,
    margin_requirement: float,
    tickers: list[str]
) -> dict:
    """Create standardized portfolio structure."""
    
    return {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0
            }
            for ticker in tickers
        }
    }
```

---

## Graph Workflow

### State Management

**File:** `src/graph/state.py`

#### `AgentState` TypedDict
Core state structure with merge operators.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]

def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    """Merge dictionaries with b taking precedence."""
    return {**a, **b}
```

#### State Flow

```
Start Node
    ↓
Selected Agents (Parallel)
    ↓
Risk Management Agent
    ↓
Portfolio Manager
    ↓
End
```

### Node Functions

#### Start Node
```python
def start(state: AgentState):
    """Initialize workflow with input data."""
    return state
```

#### Agent Nodes
Each agent node processes the state and adds analysis:
- Extracts tickers and date range
- Fetches required financial data
- Performs agent-specific analysis
- Updates analyst_signals in state
- Returns updated state

#### Risk Management Node
```python
def risk_management_agent(state: AgentState):
    """Apply risk management rules to agent signals."""
    
    # Analyze portfolio risk
    # Apply position sizing rules
    # Filter high-risk recommendations
    # Update state with risk-adjusted signals
    
    return updated_state
```

#### Portfolio Manager Node
```python
def portfolio_management_agent(state: AgentState):
    """Generate final trading decisions."""
    
    # Aggregate agent signals
    # Apply portfolio construction rules
    # Generate specific trade instructions
    # Calculate position sizes
    
    return final_decisions
```

---

## Component Integration Patterns

### Data Flow Pattern

```python
# 1. External Data → Cache → Agents
financial_data = get_financial_metrics(ticker, end_date)  # API call
cached_data = cache.get_or_set(key, financial_data)      # Caching
agent_analysis = agent.analyze(cached_data)              # Analysis

# 2. Agent Analysis → LLM → Structured Output
context = prepare_context(agent_analysis)                # Context prep
llm_response = call_llm(prompt, context)                # LLM call
structured_output = parse_response(llm_response)         # Parsing

# 3. Individual Signals → Risk Management → Final Decisions
all_signals = aggregate_agent_signals(state)            # Aggregation
risk_adjusted = apply_risk_rules(all_signals)           # Risk management
final_trades = generate_trades(risk_adjusted)           # Trade generation
```

### Error Handling Pattern

```python
def robust_agent_function(state: AgentState) -> dict:
    """Agent with comprehensive error handling."""
    
    try:
        # Core analysis logic
        result = perform_analysis(state)
        
    except DataFetchError as e:
        # Handle data unavailability
        result = create_default_signal(reason=f"Data error: {e}")
        
    except AnalysisError as e:
        # Handle analysis failures
        result = create_neutral_signal(reason=f"Analysis error: {e}")
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error in agent: {e}")
        result = create_safe_fallback_signal()
    
    finally:
        # Always update progress
        progress.update_status("agent_name", None, "Complete")
    
    return result
```

### Configuration Pattern

```python
class AgentConfig:
    """Agent configuration management."""
    
    def __init__(self):
        self.model_name = "gpt-4o"
        self.model_provider = "OpenAI"
        self.analysis_depth = "standard"
        self.risk_tolerance = "moderate"
    
    def get_model(self):
        """Get configured LLM model."""
        return get_model(self.model_name, self.model_provider)
    
    def apply_to_state(self, state: AgentState):
        """Apply configuration to state metadata."""
        state["metadata"].update({
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "analysis_depth": self.analysis_depth
        })
```

This component reference provides detailed implementation patterns and structures for extending or modifying the AI Hedge Fund system components.