# Robo-Advisor with Python - Complete System

A consolidated, fully functional Python robo-advisor combining all the features from the robo-advisor-with-python book.

## Features

1. **Mean-Variance Optimization** - Modern Portfolio Theory implementation with cvxpy
2. **Hierarchical Risk Parity (HRP)** - Machine learning-based portfolio construction
3. **Tax Loss Harvesting** - Identify and execute tax loss harvesting opportunities
4. **Portfolio Rebalancing & Backtesting** - Test strategies with historical data
5. **Tax-Efficient Withdrawal Strategies** - Optimize retirement withdrawals
6. **Monte Carlo Simulations** - 3 popular simulation types:
   - **Portfolio Value Projection** - Simulate future portfolio values
   - **Value at Risk (VaR)** - Estimate potential losses at confidence levels
   - **Retirement Sustainability** - Probability of portfolio depletion

## Installation

```bash
# Clone or navigate to the project directory
cd /Users/tux/claudestuff/robo-advisor

# Install required packages
pip install numpy pandas matplotlib yfinance cvxpy scipy

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib yfinance cvxpy scipy
```

## Quick Start

### Run the Demo

```bash
python robo_advisor.py
```

This will run 6 demonstrations:
1. Mean-Variance Optimization
2. Hierarchical Risk Parity
3. Tax Loss Harvesting
4. Portfolio Backtesting
5. Retirement Withdrawal Strategies
6. Complete Workflow Integration

## Usage Guide

### 1. Mean-Variance Optimization

```python
from robo_advisor import *

# Define assets and expected returns
tickers = ['VTI', 'VEA', 'VWO', 'AGG', 'BNDX', 'EMB']
expected_returns = pd.Series([0.05, 0.05, 0.07, 0.03, 0.02, 0.04], tickers)

# Define covariance matrix (from historical data)
covariance = pd.DataFrame([
    [0.0287, 0.0250, 0.0267, 0.0000, 0.0002, 0.0084],
    [0.0250, 0.0281, 0.0288, 0.0003, 0.0002, 0.0092],
    [0.0267, 0.0288, 0.0414, 0.0005, 0.0004, 0.0112],
    [0.0000, 0.0003, 0.0005, 0.0017, 0.0008, 0.0019],
    [0.0002, 0.0002, 0.0004, 0.0008, 0.0010, 0.0011],
    [0.0084, 0.0092, 0.0112, 0.0019, 0.0011, 0.0083]
], tickers, tickers)

# Create constraints
constraints = [
    LongOnlyConstraint(),
    FullInvestmentConstraint(),
    VolatilityConstraint(tickers, covariance, 0.10)  # 10% vol target
]

# Optimize for maximum return
optimizer = MaxExpectedReturnOptimizer(tickers, constraints, expected_returns)
optimizer.solve()
weights = optimizer.get_weights()
print(weights)
```

### 2. Hierarchical Risk Parity (HRP)

HRP is a machine learning approach that doesn't require expected returns:

```python
from robo_advisor import *
import yfinance as yf

# Download historical prices
tickers = ['VTI', 'VEA', 'VWO', 'AGG']
prices = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']

# Calculate returns
daily_returns = prices.pct_change().dropna()

# Calculate covariance and correlation
annual_cov = daily_returns.cov() * 252
correlation = daily_returns.corr()

# Calculate HRP weights
weights = hierarchical_risk_parity(annual_cov, correlation)
print("HRP Portfolio Weights:")
print(weights.round(4))

# Apply to $100,000 portfolio
portfolio_value = 100000
allocation = weights * portfolio_value
for ticker, amount in allocation.items():
    print(f"{ticker}: ${amount:,.2f} ({weights[ticker]:.1%})")
```

### 3. Tax Loss Harvesting

```python
from robo_advisor import *
import datetime as dt

# Define your tax lots
lots_data = [
    {'ticker': 'VTI', 'purchase_price': 220, 'quantity': 100,
     'purchase_date': '2023-01-15', 'sell_price': None, 'sell_date': None},
    {'ticker': 'VTI', 'purchase_price': 210, 'quantity': 50,
     'purchase_date': '2023-06-15', 'sell_price': None, 'sell_date': None},
]
lots = pd.DataFrame(lots_data)

# Define ETF replacement sets (similar ETFs that can replace each other)
etf_sets = [
    ['VTI', 'SCHB', 'ITOT'],  # US Total Stock Market
    ['VEA', 'SCHF', 'VEU'],  # Developed International
    ['VWO', 'IEMG', 'SCHE']  # Emerging Markets
]

# Current market prices
current_prices = pd.Series({'VTI': 200, 'VEA': 48, 'VWO': 40})
current_date = dt.date.today()

# Identify harvest opportunities
harvester = TaxLossHarvester(etf_sets)
opportunities = harvester.identify_harvest_opportunities(
    lots, current_prices, current_date
)

if len(opportunities) > 0:
    print(f"Found {len(opportunities)} harvest opportunities:")
    print(opportunities[['ticker', 'loss', 'replacement_options']])
else:
    print("No harvestable losses found.")
```

### 4. Portfolio Backtesting

```python
from robo_advisor import *
import yfinance as yf

# Download historical data
tickers = ['VTI', 'AGG']  # Stocks and Bonds
start_date = '2020-01-01'
end_date = '2023-12-31'

prices = yf.download(tickers, start=start_date, end=end_date)['Close']
prices.index = pd.Index(map(lambda x: x.date(), prices.index))

# Define target allocation
target_weights = pd.Series([0.6, 0.4], tickers)  # 60/40 portfolio

# Create threshold rebalancer (rebalance when deviation > 5%)
rebalancer = ThresholdRebalancer(target_weights, threshold=0.05)

# Run backtest
backtest = Backtest(prices, cash_buffer=0.02)  # Keep 2% cash
results = backtest.run(
    target_weights=target_weights,
    start_date=start_date,
    end_date=end_date,
    starting_investment=100000,
    rebalancer=rebalancer
)

# Analyze results
print(f"Starting Value: $100,000")
print(f"Ending Value: ${results['portfolio_value'].iloc[-1]:,.2f}")

total_return = (results['portfolio_value'].iloc[-1] / 100000) - 1
print(f"Total Return: {total_return:.2%}")

# Plot results
import matplotlib.pyplot as plt
results['portfolio_value'].plot(figsize=(10, 6))
plt.title('Portfolio Value Over Time')
plt.ylabel('Value ($)')
plt.grid(True)
plt.show()
```

### 5. Tax-Efficient Withdrawal Strategies

```python
from robo_advisor import *

# Setup tax calculator (2023 single filer brackets)
tax_calc = TaxCalculator()

# Define accounts and returns
returns = {
    'taxable': 0.03,  # 3% after-tax return
    'IRA': 0.04,      # 4% tax-deferred
    'Roth': 0.04      # 4% tax-free
}

starting_values = {
    'taxable': 2000000,  # $2M taxable
    'IRA': 1000000,      # $1M traditional IRA
    'Roth': 0            # $0 Roth
}

annual_spending = 120000  # $120k/year spending

# Create planner
planner = RetirementWithdrawalPlanner(
    tax_calc, returns, starting_values, annual_spending
)

# Compare different withdrawal strategies
strategies = [
    ('IRA First', ['IRA', 'taxable', 'Roth']),
    ('Taxable First', ['taxable', 'IRA', 'Roth']),
    ('Roth First', ['Roth', 'taxable', 'IRA']),
]

print("Sustainability Analysis:")
for name, order in strategies:
    years = planner.calculate_sustainability(order, max_years=50)
    print(f"  {name}: {years:.1f} years")

# Detailed simulation
df = planner.simulate_withdrawals(
    ['taxable', 'IRA', 'Roth'],  # Withdrawal order
    years=30,
    ira_fill_amount=50000,      # Fill IRA bracket to $50k
    do_roth_conversion=True      # Convert IRA to Roth
)
print("\n30-Year Projection:")
print(df.head(10).round(0).astype(int))
```

### 6. Complete Workflow

Putting it all together:

```python
from robo_advisor import *
import yfinance as yf

# Step 1: Get historical data
tickers = ['VTI', 'VEA', 'VWO', 'AGG']
prices = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Close']
returns = prices.pct_change().dropna()

# Step 2: Calculate HRP weights
cov = returns.cov() * 252
corr = returns.corr()
weights = hierarchical_risk_parity(cov, corr)
print("Target Allocation:")
for t, w in weights.items():
    print(f"  {t}: {w:.1%}")

# Step 3: Run backtest to validate
rebalancer = ThresholdRebalancer(weights, threshold=0.05)
backtest = Backtest(prices)
results = backtest.run(
    weights, '2020-01-01', '2024-01-01', 100000, rebalancer
)

# Step 4: Check tax loss harvesting opportunities
# (In real use, this would be your actual holdings)

# Step 5: Plan for retirement withdrawals
# (See section 5 above)

print(f"\nBacktest Results:")
print(f"  Total Return: {(results['portfolio_value'].iloc[-1]/100000 - 1):.1%}")
print(f"  Annualized: {((results['portfolio_value'].iloc[-1]/100000) ** (1/4) - 1):.1%}")
```

## API Reference

### Classes

#### Mean-Variance Optimization
- `MeanVarianceOptimizer` - Base optimizer class
- `MaxExpectedReturnOptimizer` - Optimize for maximum expected return
- `MinVarianceOptimizer` - Optimize for minimum variance
- `Constraint` - Base constraint class
  - `LongOnlyConstraint` - All weights >= 0
  - `FullInvestmentConstraint` - Sum of weights = 1
  - `VolatilityConstraint` - Portfolio volatility <= target
  - `MaxWeightConstraint` - Individual weights <= limit

#### Hierarchical Risk Parity
- `hierarchical_risk_parity(cov, corr)` - Calculate HRP weights

#### Tax Loss Harvesting
- `TaxLossHarvester(etf_sets)` - Identify TLH opportunities
- `check_asset_for_restrictions(lots, price, date)` - Check wash sale rules

#### Portfolio Management
- `Rebalancer` - Base rebalancer class
- `SimpleRebalancer` - Rebalance to target weights
- `ThresholdRebalancer` - Rebalance when thresholds breached
- `Backtest` - Backtesting engine

#### Tax Planning
- `TaxCalculator` - Calculate taxes with brackets
- `RetirementWithdrawalPlanner` - Optimize withdrawal strategies

### Functions

- `get_prices(assets, start_date, end_date)` - Download historical prices
- `get_dividends(assets)` - Download dividend history
- `pull_etf_returns(tickers, period)` - Download ETF returns
- `generate_efficient_frontier(returns, cov, target_vols)` - Generate efficient frontier

## Dependencies

- numpy - Numerical computing
- pandas - Data manipulation
- matplotlib - Plotting
- yfinance - Yahoo Finance data
- cvxpy - Convex optimization (optional but recommended)
- scipy - Scientific computing

## Notes

- This is an educational tool. Always consult with a financial advisor before making investment decisions.
- Historical performance does not guarantee future results.
- Tax laws change frequently; verify current tax brackets and rules.
- The backtesting engine uses simplified transaction cost models.

## License

Based on the robo-advisor-with-python book examples.
