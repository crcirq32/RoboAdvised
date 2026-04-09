#!/usr/bin/env python3
"""
Robo-Advisor with Python - Complete Portfolio Management System
===============================================================

A comprehensive robo-advisor combining:
- Mean-Variance Optimization (Modern Portfolio Theory)
- Hierarchical Risk Parity (HRP)
- Tax Loss Harvesting (TLH)
- Portfolio Rebalancing & Backtesting
- Tax-Efficient Withdrawal Strategies
- Monte Carlo Simulations (3 types: Portfolio Projection, VaR, Retirement)

Requirements:
    pip install numpy pandas matplotlib yfinance cvxpy scipy requests beautifulsoup4

Usage:
    python robo_advisor.py

Author: Consolidated from robo-advisor-with-python book
"""

import datetime as dt
import itertools
import warnings
from typing import Dict, List, Tuple, Union, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA RETRIEVAL & UTILITIES
# =============================================================================

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Using sample data.")


def get_prices(assets: List[str],
               start_date: str,
               end_date: str) -> pd.DataFrame:
    """Retrieve historical prices for given assets.

    :param assets: list of tickers
    :param start_date: first date to get prices for
    :param end_date: last date to get prices for
    :return: DataFrame of prices - one asset per column, one day per row
    """
    if YFINANCE_AVAILABLE:
        prices = yf.download(assets, start=start_date, end=end_date, progress=False)['Close']
        prices.index = pd.Index(map(lambda x: x.date(), prices.index))
        if isinstance(prices, pd.Series):
            prices = pd.DataFrame(prices)
            prices.columns = assets
        return prices
    else:
        # Generate sample price data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = pd.DataFrame(index=dates)
        np.random.seed(42)
        for asset in assets:
            prices[asset] = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
        prices.index = pd.Index(map(lambda x: x.date(), prices.index))
        return prices


def get_dividends(assets: List[str]) -> Dict:
    """Get all historical dividends for a set of assets.

    :param assets: list of tickers
    :return: dictionary keyed by ticker, with series of dividend values
    """
    div_dict = {}
    if YFINANCE_AVAILABLE:
        for ticker in assets:
            try:
                t = yf.Ticker(ticker)
                divs = t.dividends
                if len(divs) > 0:
                    divs.index = pd.Index(map(lambda x: x.date(), divs.index))
                    div_dict[ticker] = divs
            except Exception:
                div_dict[ticker] = pd.Series(dtype=float)
    else:
        # Empty dividends for sample data
        for ticker in assets:
            div_dict[ticker] = pd.Series(dtype=float)
    return div_dict


def pull_etf_returns(tickers: List[str], period: str = 'max') -> pd.DataFrame:
    """Pull historical returns for ETFs.

    :param tickers: list of ETF tickers
    :param period: time period to fetch
    :return: DataFrame of returns
    """
    if YFINANCE_AVAILABLE:
        rets = yf.download(tickers, period=period, progress=False)['Adj Close'].pct_change()
        rets = rets.dropna(axis=0, how='any')[tickers]
        return rets
    else:
        # Generate sample returns
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252*5, freq='D')
        rets = pd.DataFrame(np.random.randn(len(dates), len(tickers)) * 0.02,
                           index=dates, columns=tickers)
        return rets


# =============================================================================
# SECTION 2: MEAN-VARIANCE OPTIMIZATION
# =============================================================================

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: cvxpy not available. Optimization features will be limited.")


class Constraint:
    """Base class for portfolio constraints."""

    def generate_constraint(self, variables: Dict):
        """Create the cvxpy Constraint.

        :param variables: dictionary containing cvxpy Variables
        :return: A cvxpy Constraint object
        """
        pass


class TrackingErrorConstraint(Constraint):
    """Constraint on tracking error between portfolio and target weights."""

    def __init__(self, asset_names: Union[List[str], pd.Index],
                 reference_weights: pd.Series, sigma: pd.DataFrame, upper_bound: float):
        self.reference_weights = reference_weights.reindex(asset_names).fillna(0)
        self.sigma = sigma
        self.upper_bound = upper_bound ** 2

    def generate_constraint(self, variables: Dict):
        if not CVXPY_AVAILABLE:
            return None
        w = variables['w']
        tv = cp.quad_form(w - self.reference_weights, self.sigma)
        return tv <= self.upper_bound


class VolatilityConstraint(TrackingErrorConstraint):
    """Constraint on overall portfolio volatility."""

    def __init__(self, asset_names: Union[List[str], pd.Index],
                 sigma: pd.DataFrame, upper_bound: float):
        zeros = pd.Series(np.zeros(len(asset_names)), asset_names)
        super().__init__(asset_names, zeros, sigma, upper_bound)


class LinearConstraint(Constraint):
    """Generic linear constraint of form: coefs * w [vs] rhs"""

    def __init__(self, asset_names: List[str], coefs: pd.Series,
                 rhs: float, direction: str):
        self.coefs = coefs.reindex(asset_names).fillna(0).values
        self.rhs = rhs
        self.direction = direction

    def generate_constraint(self, variables: Dict):
        if not CVXPY_AVAILABLE:
            return None
        w = variables['w']
        direction = self.direction
        if direction[0] == '<':
            return self.coefs.T @ w <= self.rhs
        elif direction[0] == '>':
            return self.coefs.T @ w >= self.rhs
        elif direction[0] == '=':
            return self.coefs.T @ w == self.rhs


class LongOnlyConstraint(Constraint):
    """Constraint to enforce all portfolio weights are non-negative."""

    def generate_constraint(self, variables: Dict):
        if not CVXPY_AVAILABLE:
            return None
        return variables['w'] >= 0


class FullInvestmentConstraint(Constraint):
    """Constraint to enforce sum of portfolio weights is one."""

    def generate_constraint(self, variables: Dict):
        if not CVXPY_AVAILABLE:
            return None
        return cp.sum(variables['w']) == 1.0


class MaxWeightConstraint(Constraint):
    """Upper bound on magnitude of every asset in portfolio."""

    def __init__(self, upper_bound: float):
        self.upper_bound = upper_bound

    def generate_constraint(self, variables: Dict):
        if not CVXPY_AVAILABLE:
            return None
        return cp.norm_inf(variables['w']) <= self.upper_bound


class MeanVarianceOptimizer:
    """Mean-Variance Portfolio Optimizer using cvxpy."""

    def __init__(self):
        self.asset_names = []
        self.variables = None
        self.prob = None

    def solve(self):
        """Solve the optimization problem."""
        if CVXPY_AVAILABLE and self.prob is not None:
            self.prob.solve()

    def get_weights(self) -> pd.Series:
        """Get optimized portfolio weights."""
        if CVXPY_AVAILABLE and self.variables is not None:
            return pd.Series(self.variables['w'].value, self.asset_names)
        return pd.Series(1.0/len(self.asset_names), self.asset_names)


class MaxExpectedReturnOptimizer(MeanVarianceOptimizer):
    """Optimizer for maximum expected return subject to constraints."""

    def __init__(self, asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint], expected_returns: pd.Series):
        super().__init__()
        if not CVXPY_AVAILABLE:
            self.asset_names = asset_names
            return

        self.asset_names = asset_names
        variables = {'w': cp.Variable(len(expected_returns))}

        cons = [c.generate_constraint(variables) for c in constraints if c.generate_constraint(variables) is not None]
        obj = cp.Maximize(expected_returns.values.T @ variables['w'])
        self.variables = variables
        self.prob = cp.Problem(obj, cons)


class MinVarianceOptimizer(MeanVarianceOptimizer):
    """Optimizer for minimum variance portfolio."""

    def __init__(self, asset_names: Union[List[str], pd.Index],
                 constraints: List[Constraint], sigma: pd.DataFrame):
        super().__init__()
        if not CVXPY_AVAILABLE:
            self.asset_names = asset_names
            return

        self.asset_names = asset_names
        variables = {'w': cp.Variable(len(asset_names))}

        cons = [c.generate_constraint(variables) for c in constraints if c.generate_constraint(variables) is not None]
        obj = cp.Minimize(cp.quad_form(variables['w'], sigma.values))
        self.variables = variables
        self.prob = cp.Problem(obj, cons)


def generate_efficient_frontier(expected_returns: pd.Series, sigma: pd.DataFrame,
                                 target_vols: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Generate the efficient frontier.

    :param expected_returns: expected returns for each asset
    :param sigma: covariance matrix
    :param target_vols: target volatilities (optional)
    :return: DataFrame with efficient frontier points
    """
    asset_vols = np.sqrt(np.diag(sigma))
    if target_vols is None:
        target_vols = np.arange(np.min(asset_vols) + 0.005,
                                np.max(asset_vols) + 0.02, 0.005)

    results = []
    for target_vol in target_vols:
        cons = [LongOnlyConstraint(), FullInvestmentConstraint(),
                VolatilityConstraint(expected_returns.index, sigma, target_vol)]

        opt = MaxExpectedReturnOptimizer(expected_returns.index, cons, expected_returns)
        opt.solve()
        weights = opt.get_weights()

        if weights.isna().any():
            continue

        risk = np.sqrt(weights @ sigma @ weights)
        er = weights @ expected_returns
        if risk < (target_vol - 0.005):
            continue

        info = pd.Series([risk, er], ['Risk', 'Expected_Return'])
        results.append(pd.concat((info, weights)))

    if results:
        return pd.concat(results, axis=1).T
    return pd.DataFrame()


# =============================================================================
# SECTION 3: HIERARCHICAL RISK PARITY (HRP)
# =============================================================================

def calc_distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Transform correlation matrix to distance matrix.

    :param corr: correlation matrix
    :return: distance matrix
    """
    return ((1 - corr) / 2.) ** 0.5


def calc_linkage_matrix(d: pd.DataFrame) -> np.ndarray:
    """Calculate hierarchical linkage matrix using single linkage.

    :param d: distance matrix
    :return: linkage matrix for hierarchical clustering
    """
    from scipy.cluster.hierarchy import linkage
    # Convert distance matrix to condensed form
    n = d.shape[0]
    idx = np.triu_indices(n, k=1)
    condensed = d.values[idx]
    return linkage(condensed, method='single')


def get_quasi_diagonal_order(link_mat: np.ndarray, tickers: List[str]) -> pd.Series:
    """Get quasi-diagonal ordering from linkage matrix.

    :param link_mat: linkage matrix
    :param tickers: asset tickers
    :return: ordered series of tickers
    """
    from scipy.cluster.hierarchy import leaves_list
    order_indices = leaves_list(link_mat)
    return pd.Series([tickers[i] for i in order_indices])


def split_indices(indices: List[pd.Index]) -> List[pd.Index]:
    """Split indices into left and right halves."""
    splits = []
    for i in indices:
        if len(i) <= 1:
            continue
        mid = len(i) // 2
        splits.extend([i[:mid], i[mid:]])
    return splits


def calc_cluster_variance(cov: pd.DataFrame, assets: List[str]) -> float:
    """Calculate variance of equally-weighted cluster.

    :param cov: covariance matrix
    :param assets: assets in cluster
    :return: cluster variance
    """
    if len(assets) == 0:
        return 0.0
    sub_cov = cov.loc[assets, assets]
    w = (1.0 / np.diag(sub_cov)).reshape(-1, 1)
    w /= w.sum()
    variance = float((w.T @ sub_cov.values @ w)[0, 0])
    return variance if not np.isnan(variance) else 0.0


def calc_hrp_weights(cov: pd.DataFrame, ordering: pd.Series) -> pd.Series:
    """Calculate Hierarchical Risk Parity weights.

    :param cov: covariance matrix
    :param ordering: ordered assets
    :return: HRP weights
    """
    weights = pd.Series(1.0, index=ordering.values)
    indices = [pd.RangeIndex(len(ordering))]

    while len(indices) > 0:
        indices = split_indices(indices)
        for i in range(0, len(indices), 2):
            if i + 1 >= len(indices):
                break
            i_left, i_right = indices[i], indices[i + 1]

            left_assets = [ordering.values[idx] for idx in i_left]
            right_assets = [ordering.values[idx] for idx in i_right]

            left_var = calc_cluster_variance(cov, left_assets)
            right_var = calc_cluster_variance(cov, right_assets)

            if left_var + right_var == 0 or np.isnan(left_var + right_var):
                alpha = 0.5
            else:
                alpha = left_var / (left_var + right_var)

            for idx in i_left:
                weights.loc[ordering.values[idx]] *= (1 - alpha)
            for idx in i_right:
                weights.loc[ordering.values[idx]] *= alpha

    return weights


def hierarchical_risk_parity(cov: pd.DataFrame, corr: pd.DataFrame) -> pd.Series:
    """Calculate Hierarchical Risk Parity portfolio weights.

    :param cov: asset-level covariance matrix
    :param corr: asset-level correlation matrix
    :return: HRP portfolio weights
    """
    d = calc_distance_matrix(corr)
    tickers = list(d.columns)
    link_mat = calc_linkage_matrix(d)
    ordering = get_quasi_diagonal_order(link_mat, tickers)
    weights = calc_hrp_weights(cov, ordering)
    return weights.reindex(tickers)


# =============================================================================
# SECTION 4: TAX LOSS HARVESTING
# =============================================================================

def sellable(lot: pd.Series, new_lots: pd.Series) -> pd.Series:
    """Check whether a lot can be sold without creating a wash sale.

    :param lot: Information about the tax lot
    :param new_lots: Series containing purchase dates of newly-purchased lots
    :return: Series with sellable flag and blocking lots
    """
    idx = ['sellable', 'blocking_lots']
    if not lot['still_held']:
        return pd.Series([False, pd.Series(dtype='str')], idx)

    if not lot['is_at_loss']:
        return pd.Series([True, pd.Series(dtype='str')], idx)

    if lot['is_new'] and len(new_lots) > 1:
        blocking_lots = new_lots[new_lots != lot.purchase_date]
        return pd.Series([False, blocking_lots], idx)

    if not lot['is_new'] and len(new_lots) > 0:
        return pd.Series([False, new_lots], idx)

    return pd.Series([True, pd.Series(dtype='str')], idx)


def check_asset_for_restrictions(lots: pd.DataFrame, current_price: float,
                                  current_date: dt.date) -> pd.DataFrame:
    """Check buying and selling eligibility for an asset.

    :param lots: tax lots
    :param current_price: current asset price
    :param current_date: current date
    :return: Original lots with wash sale information appended
    """
    lots = lots.copy()
    ws_start = (current_date - dt.timedelta(days=30)).strftime('%Y-%m-%d')
    lots['is_new'] = list(map(lambda x: x >= ws_start, lots['purchase_date']))
    lots['is_at_loss'] = list(map(lambda x: current_price < x, lots['purchase_price']))
    lots['still_held'] = list(map(lambda x: x != x, lots['sell_date']))

    new_lots = lots[lots['is_new'] & lots['still_held']]['purchase_date']
    sellability = {i: sellable(lots.loc[i], new_lots) for i in lots.index}
    lots = lots.join(pd.DataFrame(sellability).T)

    buy_blocks = [blocks_buying(row[1], current_date) for row in lots.iterrows()]
    lots['blocks_buy'] = pd.Series(buy_blocks, lots.index)
    lots.drop(['is_new', 'is_at_loss', 'still_held'], axis=1, inplace=True)

    return lots


def blocks_buying(lot: pd.Series, current_date: dt.date) -> bool:
    """Check if lot prevents buying due to recent loss sale."""
    if lot['still_held']:
        return False
    how_long = (current_date - dt.date.fromisoformat(lot['sell_date'])).days
    return how_long <= 30 and lot['sell_price'] < lot['purchase_price']


class TaxLossHarvester:
    """Tax Loss Harvesting engine."""

    def __init__(self, etf_replacement_sets: List[List[str]]):
        """
        :param etf_replacement_sets: List of ETF groups that can replace each other
                                    e.g., [['VTI', 'SCHB', 'ITOT'], ['VEA', 'SCHF']]
        """
        self.etf_sets = etf_replacement_sets

    def identify_harvest_opportunities(self, lots: pd.DataFrame,
                                       current_prices: pd.Series,
                                       current_date: dt.date) -> pd.DataFrame:
        """Identify lots that can be harvested for tax losses.

        :param lots: DataFrame of tax lots
        :param current_prices: Current asset prices
        :param current_date: Current date
        :return: DataFrame of harvest opportunities
        """
        opportunities = []

        for etf_set in self.etf_sets:
            set_lots = lots[lots['ticker'].isin(etf_set)]
            if len(set_lots) == 0:
                continue

            # Check restrictions for each asset
            for ticker in etf_set:
                asset_lots = set_lots[set_lots['ticker'] == ticker].copy()
                if len(asset_lots) == 0:
                    continue

                asset_lots = check_asset_for_restrictions(
                    asset_lots, current_prices.get(ticker, 0), current_date
                )

                # Find sellable lots at a loss
                for idx, lot in asset_lots.iterrows():
                    if lot.get('sellable', False):
                        current_price = current_prices.get(ticker, 0)
                        gain = (current_price - lot['purchase_price']) * lot['quantity']

                        if gain < 0:  # Loss
                            opp = {
                                'ticker': ticker,
                                'purchase_date': lot['purchase_date'],
                                'purchase_price': lot['purchase_price'],
                                'current_price': current_price,
                                'quantity': lot['quantity'],
                                'loss': -gain,
                                'replacement_options': [e for e in etf_set if e != ticker]
                            }
                            opportunities.append(opp)

        return pd.DataFrame(opportunities) if opportunities else pd.DataFrame()


# =============================================================================
# SECTION 5: PORTFOLIO REBALANCING & BACKTESTING
# =============================================================================

class Rebalancer:
    """Base class for portfolio rebalancers."""

    def __init__(self, target_weights: pd.Series):
        self.target_weights = target_weights

    def rebalance(self, date: dt.date, holdings: pd.DataFrame,
                  investment_value: float) -> Dict:
        """Generate rebalance trades.

        :param date: current date
        :param holdings: current holdings
        :param investment_value: total portfolio value
        :return: dictionary with 'buys' and 'sells'
        """
        pass


class SimpleRebalancer(Rebalancer):
    """Simple rebalancer that trades back to target weights."""

    def __init__(self, target_weights: pd.Series, tax_params: Optional[Dict] = None):
        super().__init__(target_weights)
        self.tax_params = tax_params or {'lt_gains_rate': 0.15, 'income_rate': 0.22, 'lt_cutoff': 365}

    def rebalance(self, date: dt.date, holdings: pd.DataFrame,
                  investment_value: float) -> Dict:
        """Generate trades to rebalance to target weights."""
        asset_holdings = holdings[['ticker', 'value']].groupby(['ticker']).sum()['value'] if len(holdings) > 0 else pd.Series()
        target_values = self.target_weights * investment_value

        full_index = asset_holdings.index.union(target_values.index)
        trade_values = target_values.reindex(full_index).fillna(0) - asset_holdings.reindex(full_index).fillna(0)

        buys = trade_values[trade_values > 0]
        sells = trade_values[trade_values < 0]

        return {'buys': buys, 'sells': sells}


class ThresholdRebalancer(SimpleRebalancer):
    """Rebalancer that triggers when deviation exceeds threshold."""

    def __init__(self, target_weights: pd.Series, threshold: float, tax_params: Optional[Dict] = None):
        super().__init__(target_weights, tax_params)
        self.threshold = threshold

    def rebalance(self, date: dt.date, holdings: pd.DataFrame,
                  investment_value: float) -> Dict:
        """Rebalance only if any asset deviates beyond threshold."""
        if len(holdings) == 0:
            return super().rebalance(date, holdings, investment_value)

        current_weights = holdings[['ticker', 'value']].groupby(['ticker']).sum()['value'] / investment_value
        deviations = (current_weights - self.target_weights).abs()

        if deviations.max() < self.threshold:
            return {'buys': pd.Series(dtype=float), 'sells': pd.Series(dtype=float)}

        return super().rebalance(date, holdings, investment_value)


class Backtest:
    """Portfolio backtesting engine."""

    def __init__(self, prices: pd.DataFrame, dividends: Optional[Dict] = None,
                 cash_buffer: float = 0.02):
        """
        :param prices: DataFrame of asset prices
        :param dividends: Dictionary of dividend series
        :param cash_buffer: percentage of portfolio to keep as cash
        """
        self.prices = prices
        self.dividends = dividends or {}
        self.cash_buffer = cash_buffer

    def run(self, target_weights: pd.Series, start_date: str, end_date: str,
            starting_investment: float, rebalancer: Rebalancer) -> pd.DataFrame:
        """Run backtest.

        :param target_weights: target portfolio weights
        :param start_date: start date string
        :param end_date: end date string
        :param starting_investment: initial investment amount
        :param rebalancer: rebalancer instance
        :return: DataFrame with daily results
        """
        cash = starting_investment
        holdings = pd.DataFrame(columns=['ticker', 'value', 'quantity', 'purchase_price', 'purchase_date'])

        # Filter prices to date range
        mask = (self.prices.index >= pd.Timestamp(start_date).date()) & \
               (self.prices.index <= pd.Timestamp(end_date).date())
        prices = self.prices.loc[mask]

        results = []

        for date in prices.index:
            current_prices = prices.loc[date]

            # Mark to market
            if len(holdings) > 0:
                holdings['current_price'] = holdings['ticker'].map(current_prices)
                holdings['value'] = holdings['current_price'] * holdings['quantity']

            # Calculate dividends
            div_income = self._calc_dividends(date, holdings)
            cash += div_income

            portfolio_value = holdings['value'].sum() + cash
            investment_value = portfolio_value * (1 - self.cash_buffer)

            # Get rebalance trades
            trades = rebalancer.rebalance(date, holdings, investment_value)

            # Execute trades
            holdings, cash = self._execute_trades(date, holdings, trades, current_prices, cash)

            # Record results
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'invested_value': investment_value,
                'dividends': div_income
            })

        return pd.DataFrame(results).set_index('date')

    def _calc_dividends(self, date: dt.date, holdings: pd.DataFrame) -> float:
        """Calculate dividend income for the day."""
        if len(holdings) == 0:
            return 0.0

        div_income = 0.0
        shares_by_asset = holdings.groupby('ticker')['quantity'].sum()

        for asset in shares_by_asset.index:
            if asset in self.dividends:
                try:
                    div = self.dividends[asset].get(date, 0)
                    div_income += shares_by_asset[asset] * div
                except:
                    pass

        return div_income

    def _execute_trades(self, date: dt.date, holdings: pd.DataFrame,
                        trades: Dict, prices: pd.Series, cash: float) -> Tuple[pd.DataFrame, float]:
        """Execute buy/sell trades."""
        holdings = holdings.copy()

        # Process sells
        sells = trades.get('sells', pd.Series())
        for ticker, amount in sells.items():
            if ticker in prices and amount > 0:
                price = prices[ticker]
                shares_to_sell = amount / price

                # Find lots to sell from
                ticker_lots = holdings[holdings['ticker'] == ticker]
                remaining = shares_to_sell

                for idx in ticker_lots.index:
                    if remaining <= 0:
                        break
                    lot_qty = holdings.loc[idx, 'quantity']
                    sell_qty = min(lot_qty, remaining)
                    holdings.loc[idx, 'quantity'] -= sell_qty
                    remaining -= sell_qty

                cash += (shares_to_sell - remaining) * price

        # Process buys
        buys = trades.get('buys', pd.Series())
        for ticker, amount in buys.items():
            if ticker in prices and amount > 0:
                price = prices[ticker]
                shares = amount / price

                if cash >= amount:
                    new_lot = pd.DataFrame([{
                        'ticker': ticker,
                        'quantity': shares,
                        'purchase_price': price,
                        'purchase_date': date.strftime('%Y-%m-%d'),
                        'value': amount,
                        'current_price': price
                    }])
                    holdings = pd.concat([holdings, new_lot], ignore_index=True)
                    cash -= amount

        # Remove empty lots
        holdings = holdings[holdings['quantity'] > 1e-10]

        return holdings, cash


# =============================================================================
# SECTION 6: TAX-EFFICIENT WITHDRAWAL STRATEGIES
# =============================================================================

class TaxCalculator:
    """Tax calculation engine."""

    def __init__(self, tax_brackets: Optional[pd.DataFrame] = None,
                 standard_deduction: float = 13850):
        """
        :param tax_brackets: DataFrame with 'rate' and 'top_of_bracket' columns
        :param standard_deduction: standard deduction amount
        """
        if tax_brackets is None:
            # 2023 Federal Tax Brackets (Single)
            rates = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
            tops = [11000, 44725, 95375, 182100, 231250, 578125, np.inf]
            tax_brackets = pd.DataFrame({'rate': rates, 'top_of_bracket': tops})

        self.tax_brackets = tax_brackets
        self.standard_deduction = standard_deduction

    def calculate_tax(self, income: float) -> float:
        """Calculate tax on given income.

        :param income: taxable income
        :return: tax owed
        """
        taxable_income = max(0, income - self.standard_deduction)
        tax = 0
        bottom = 0
        i = 0

        while taxable_income > bottom and i < len(self.tax_brackets):
            bracket_top = self.tax_brackets.iloc[i]['top_of_bracket']
            bracket_income = min(bracket_top, taxable_income) - bottom
            tax += self.tax_brackets.iloc[i]['rate'] * bracket_income
            bottom = bracket_top
            i += 1

        return tax

    def gross_up(self, net_amount: float) -> float:
        """Calculate gross amount needed for desired net amount.

        :param net_amount: desired after-tax amount
        :return: gross amount needed
        """
        from scipy.optimize import fsolve

        def objective(gross):
            return gross - self.calculate_tax(gross) - net_amount

        result = fsolve(objective, net_amount * 1.3)
        return result[0]


class RetirementWithdrawalPlanner:
    """Retirement withdrawal strategy planner."""

    def __init__(self, tax_calculator: TaxCalculator,
                 returns: Dict[str, float],
                 starting_values: Dict[str, float],
                 annual_spending: float):
        """
        :param tax_calculator: TaxCalculator instance
        :param returns: dict of expected returns by account type
        :param starting_values: dict of starting balances by account type
        :param annual_spending: annual spending requirement
        """
        self.tax_calc = tax_calculator
        self.returns = returns
        self.starting_values = starting_values
        self.annual_spending = annual_spending

    def simulate_withdrawals(self, withdrawal_order: List[str],
                             years: int = 30,
                             ira_fill_amount: float = 0,
                             do_roth_conversion: bool = False) -> pd.DataFrame:
        """Simulate withdrawal strategy over time.

        :param withdrawal_order: order to withdraw from accounts
        :param years: number of years to simulate
        :param ira_fill_amount: amount to fill IRA bracket annually
        :param do_roth_conversion: whether to convert IRA to Roth
        :return: DataFrame with account balances over time
        """
        df = pd.DataFrame(0.0, columns=['taxable', 'IRA', 'Roth'], index=range(years + 1))
        df.iloc[0] = self.starting_values

        for i in range(years):
            if df.iloc[i].sum() <= 0:
                break

            spending_left = self.annual_spending

            # Handle IRA fill/conversion first if specified
            if ira_fill_amount > 0 and withdrawal_order[0] == 'taxable' and df.iloc[i]['taxable'] > self.annual_spending:
                ira_withdraw = min(df.iloc[i]['IRA'], ira_fill_amount)

                if do_roth_conversion:
                    # Convert to Roth
                    tax = self.tax_calc.calculate_tax(ira_withdraw)
                    df.iloc[i]['IRA'] -= ira_withdraw
                    df.iloc[i]['Roth'] += (ira_withdraw - tax)
                else:
                    # Regular withdrawal
                    tax = self.tax_calc.calculate_tax(ira_withdraw)
                    spending_left -= (ira_withdraw - tax)
                    df.iloc[i]['IRA'] -= ira_withdraw

            # Process withdrawal order
            for acct_type in withdrawal_order:
                if spending_left <= 0:
                    break

                if acct_type == 'IRA':
                    # Gross up for taxes
                    gross_needed = self.tax_calc.gross_up(spending_left)
                    withdraw = min(df.iloc[i][acct_type], gross_needed)
                    tax = self.tax_calc.calculate_tax(withdraw)
                    spending_left -= (withdraw - tax)
                else:
                    withdraw = min(df.iloc[i][acct_type], spending_left)
                    spending_left -= withdraw

                df.iloc[i + 1][acct_type] = (df.iloc[i][acct_type] - withdraw) * (1 + self.returns[acct_type])

        return df

    def calculate_sustainability(self, withdrawal_order: List[str],
                                  max_years: int = 50,
                                  **kwargs) -> float:
        """Calculate how many years the portfolio can sustain withdrawals.

        :param withdrawal_order: order to withdraw from accounts
        :param max_years: maximum years to simulate
        :return: years of sustainability
        """
        df = self.simulate_withdrawals(withdrawal_order, max_years, **kwargs)

        for i in range(len(df)):
            total = df.iloc[i].sum()
            if total < self.annual_spending:
                # Fractional year
                prev_total = df.iloc[i-1].sum() if i > 0 else total
                fraction = max(0, total / self.annual_spending) if prev_total > 0 else 0
                return i + fraction

        return max_years


# =============================================================================
# SECTION 6: MONTE CARLO SIMULATIONS
# =============================================================================

class MonteCarloSimulator:
    """Monte Carlo simulation engine for portfolio analysis."""

    def __init__(self, expected_returns: pd.Series, covariance: pd.DataFrame,
                 weights: Optional[pd.Series] = None, seed: Optional[int] = None):
        """
        :param expected_returns: Expected annual returns for each asset
        :param covariance: Annual covariance matrix
        :param weights: Portfolio weights (if None, equal weights assumed)
        :param seed: Random seed for reproducibility
        """
        self.expected_returns = expected_returns
        self.covariance = covariance
        self.assets = list(expected_returns.index)
        self.weights = weights if weights is not None else pd.Series(
            1/len(self.assets), index=self.assets)

        if seed is not None:
            np.random.seed(seed)

    def _portfolio_return(self) -> float:
        """Calculate portfolio expected return."""
        return (self.weights * self.expected_returns).sum()

    def _portfolio_volatility(self) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(self.weights @ self.covariance @ self.weights)

    def simulate_portfolio_values(self, initial_value: float,
                                   years: int,
                                   n_simulations: int = 1000,
                                   annual_contribution: float = 0) -> pd.DataFrame:
        """Monte Carlo Simulation #1: Portfolio Value Projection

        Simulates future portfolio values assuming log-normal returns.
        Used for retirement planning and goal-based investing.

        :param initial_value: Starting portfolio value
        :param years: Number of years to simulate
        :param n_simulations: Number of simulation paths
        :param annual_contribution: Annual contribution (optional)
        :return: DataFrame with simulation results (rows=years, columns=simulations)
        """
        mu = self._portfolio_return()
        sigma = self._portfolio_volatility()

        dt = 1  # Annual time step
        paths = np.zeros((years + 1, n_simulations))
        paths[0, :] = initial_value

        for t in range(1, years + 1):
            z = np.random.standard_normal(n_simulations)
            # Geometric Brownian Motion
            paths[t, :] = paths[t-1, :] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            ) + annual_contribution

        return pd.DataFrame(paths, columns=[f'Sim_{i}' for i in range(n_simulations)])

    def calculate_var_cvar(self, initial_value: float,
                           confidence_level: float = 0.95,
                           time_horizon_days: int = 252,
                           n_simulations: int = 10000) -> Dict:
        """Monte Carlo Simulation #2: Value at Risk (VaR) and CVaR

        Estimates potential losses at a given confidence level.
        Used for risk management and position sizing.

        :param initial_value: Current portfolio value
        :param confidence_level: Confidence level (e.g., 0.95 for 95%)
        :param time_horizon_days: Time horizon for VaR calculation
        :param n_simulations: Number of simulations
        :return: Dictionary with VaR, CVaR, and simulation results
        """
        mu = self._portfolio_return()
        sigma = self._portfolio_volatility()

        dt = time_horizon_days / 252  # Fraction of year

        # Simulate returns
        z = np.random.standard_normal(n_simulations)
        portfolio_returns = np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        ) - 1

        # Calculate losses (negative returns)
        losses = -portfolio_returns * initial_value
        sorted_losses = np.sort(losses)

        # VaR: The loss at the confidence level
        var_index = int((1 - confidence_level) * n_simulations)
        var = sorted_losses[var_index]

        # CVaR (Expected Shortfall): Average of losses beyond VaR
        cvar = sorted_losses[:var_index].mean() if var_index > 0 else var

        return {
            'VaR': var,
            'VaR_pct': var / initial_value,
            'CVaR': cvar,
            'CVaR_pct': cvar / initial_value,
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon_days,
            'simulated_losses': losses,
            'simulated_returns': portfolio_returns
        }

    def simulate_retirement_sustainability(self,
                                           starting_balance: Dict[str, float],
                                           annual_withdrawal: float,
                                           years: int,
                                           n_simulations: int = 1000,
                                           withdrawal_order: List[str] = None) -> pd.DataFrame:
        """Monte Carlo Simulation #3: Retirement Sustainability Analysis

        Simulates multiple retirement scenarios to estimate probability
        of portfolio depletion and median/percentile outcomes.
        Used for retirement income planning.

        :param starting_balance: Dict with starting values for each account type
                                e.g., {'taxable': 2000000, 'IRA': 1000000, 'Roth': 0}
        :param annual_withdrawal: Annual amount to withdraw
        :param years: Number of retirement years to simulate
        :param n_simulations: Number of simulation paths
        :param withdrawal_order: Order to withdraw from accounts (default: ['taxable', 'IRA', 'Roth'])
        :return: DataFrame with percentile outcomes over time
        """
        if withdrawal_order is None:
            withdrawal_order = ['taxable', 'IRA', 'Roth']

        # Assume same return/vol for all account types based on portfolio
        mu = self._portfolio_return()
        sigma = self._portfolio_volatility()

        dt = 1
        all_sims = []

        for sim in range(n_simulations):
            balances = {k: v for k, v in starting_balance.items()}
            path = []

            for year in range(years + 1):
                total_balance = sum(balances.values())
                path.append(total_balance)

                if total_balance <= annual_withdrawal:
                    # Portfolio depleted
                    for k in balances:
                        balances[k] = 0
                    continue

                # Withdraw based on order
                withdrawal_left = annual_withdrawal
                for acct in withdrawal_order:
                    if withdrawal_left <= 0:
                        break
                    withdraw = min(balances[acct], withdrawal_left)
                    balances[acct] -= withdraw
                    withdrawal_left -= withdraw

                # Apply market returns
                z = np.random.standard_normal()
                market_return = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z) - 1

                for acct in balances:
                    balances[acct] *= (1 + market_return)

            all_sims.append(path)

        sims_df = pd.DataFrame(all_sims).T

        # Calculate percentiles
        percentiles = pd.DataFrame({
            'Year': range(years + 1),
            'P10': sims_df.quantile(0.10, axis=1),
            'P25': sims_df.quantile(0.25, axis=1),
            'P50': sims_df.quantile(0.50, axis=1),
            'Mean': sims_df.mean(axis=1),
            'P75': sims_df.quantile(0.75, axis=1),
            'P90': sims_df.quantile(0.90, axis=1),
        })

        # Calculate success probability (not depleted)
        depleted = (sims_df.iloc[-1] <= 0).sum()
        success_rate = (n_simulations - depleted) / n_simulations

        percentiles['Success_Rate'] = success_rate

        return percentiles


def run_monte_carlo_analysis(portfolio_value: float,
                             expected_returns: pd.Series,
                             covariance: pd.DataFrame,
                             weights: pd.Series,
                             years: int = 30,
                             annual_contribution: float = 0,
                             annual_withdrawal: float = 0,
                             n_sims: int = 1000,
                             plot: bool = True) -> Dict:
    """Run complete Monte Carlo analysis with all 3 simulation types.

    :param portfolio_value: Current portfolio value
    :param expected_returns: Expected annual returns
    :param covariance: Covariance matrix
    :param weights: Portfolio weights
    :param years: Projection years
    :param annual_contribution: Annual contributions (accumulation)
    :param annual_withdrawal: Annual withdrawals (decumulation)
    :param n_sims: Number of simulations
    :param plot: Whether to generate plots
    :return: Dictionary with all simulation results
    """
    mc = MonteCarloSimulator(expected_returns, covariance, weights, seed=42)

    results = {}

    # Simulation 1: Portfolio Value Projection
    print("Running Portfolio Value Projection...")
    value_paths = mc.simulate_portfolio_values(
        portfolio_value, years, n_sims, annual_contribution
    )
    results['value_projection'] = value_paths

    # Simulation 2: VaR Analysis
    print("Running Value at Risk Analysis...")
    var_results = mc.calculate_var_cvar(portfolio_value, n_simulations=n_sims*10)
    results['var_analysis'] = var_results

    # Simulation 3: Retirement Sustainability (if withdrawals specified)
    if annual_withdrawal > 0:
        print("Running Retirement Sustainability Analysis...")
        # Assume all in taxable for simplicity
        starting = {'taxable': portfolio_value, 'IRA': 0, 'Roth': 0}
        retirement_sims = mc.simulate_retirement_sustainability(
            starting, annual_withdrawal, years, n_sims
        )
        results['retirement'] = retirement_sims

    # Summary statistics
    final_values = value_paths.iloc[-1]
    results['summary'] = {
        'median_final_value': final_values.median(),
        'mean_final_value': final_values.mean(),
        'p10_final_value': final_values.quantile(0.10),
        'p90_final_value': final_values.quantile(0.90),
        'probability_of_gain': (final_values > portfolio_value).mean(),
        'var_95': var_results['VaR'],
        'cvar_95': var_results['CVaR'],
    }

    if plot and annual_withdrawal > 0:
        # Plot retirement outcomes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Portfolio Value Projection (sample paths)
        ax1 = axes[0, 0]
        sample_paths = value_paths.iloc[:, :100]  # Show first 100 paths
        ax1.plot(sample_paths.index, sample_paths, alpha=0.1, color='blue')
        ax1.plot(value_paths.index, value_paths.median(axis=1),
                color='red', linewidth=2, label='Median')
        ax1.axhline(y=portfolio_value, color='black', linestyle='--',
                   label='Initial Value')
        ax1.set_title('Portfolio Value Projection (Monte Carlo)')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Final Value Distribution
        ax2 = axes[0, 1]
        ax2.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(final_values.median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: ${final_values.median:,.0f}')
        ax2.set_title('Distribution of Final Portfolio Values')
        ax2.set_xlabel('Final Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: VaR Distribution
        ax3 = axes[1, 0]
        losses = var_results['simulated_losses']
        ax3.hist(losses, bins=50, alpha=0.7, edgecolor='black', color='red')
        ax3.axvline(var_results['VaR'], color='darkred', linestyle='--',
                   linewidth=2, label=f"95% VaR: ${var_results['VaR']:,.0f}")
        ax3.set_title('Value at Risk (VaR) Distribution')
        ax3.set_xlabel('Potential Loss ($)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Retirement Sustainability
        ax4 = axes[1, 1]
        retirement = results.get('retirement')
        if retirement is not None:
            ax4.fill_between(retirement['Year'], retirement['P10'], retirement['P90'],
                           alpha=0.3, label='10th-90th Percentile')
            ax4.plot(retirement['Year'], retirement['P50'], color='red',
                    linewidth=2, label='Median')
            ax4.axhline(y=annual_withdrawal, color='orange', linestyle='--',
                       label=f'Annual Withdrawal (${annual_withdrawal:,.0f})')
            ax4.set_title(f'Retirement Sustainability\n'
                         f'Success Rate: {retirement["Success_Rate"].iloc[0]:.1%}')
            ax4.set_xlabel('Years in Retirement')
            ax4.set_ylabel('Portfolio Value ($)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/Users/tux/claudestuff/robo-advisor/monte_carlo_results.png')
        print("\nChart saved to: monte_carlo_results.png")
        plt.close()

    return results


# =============================================================================
# SECTION 7: DEMONSTRATION & USAGE EXAMPLES
# =============================================================================

def demo_mean_variance_optimization():
    """Demonstrate Mean-Variance Optimization."""
    print("\n" + "="*60)
    print("MEAN-VARIANCE OPTIMIZATION DEMO")
    print("="*60)

    # Sample data
    tickers = ['VTI', 'VEA', 'VWO', 'AGG', 'BNDX', 'EMB']
    expected_returns = pd.Series([0.05, 0.05, 0.07, 0.03, 0.02, 0.04], tickers)

    # Sample covariance matrix
    sigma = np.array([
        [0.0287, 0.0250, 0.0267, 0.0000, 0.0002, 0.0084],
        [0.0250, 0.0281, 0.0288, 0.0003, 0.0002, 0.0092],
        [0.0267, 0.0288, 0.0414, 0.0005, 0.0004, 0.0112],
        [0.0000, 0.0003, 0.0005, 0.0017, 0.0008, 0.0019],
        [0.0002, 0.0002, 0.0004, 0.0008, 0.0010, 0.0011],
        [0.0084, 0.0092, 0.0112, 0.0019, 0.0011, 0.0083]
    ])
    sigma = pd.DataFrame(sigma, tickers, tickers)

    print("\nExpected Returns:")
    print(expected_returns)

    print("\nCovariance Matrix:")
    print(sigma)

    if CVXPY_AVAILABLE:
        # Create minimum variance portfolio
        print("\n--- Minimum Variance Portfolio ---")
        cons = [LongOnlyConstraint(), FullInvestmentConstraint()]
        min_var = MinVarianceOptimizer(tickers, cons, sigma)
        min_var.solve()
        weights = min_var.get_weights()
        print("Optimal Weights:")
        print(weights.round(4))
        print(f"Portfolio Volatility: {np.sqrt(weights @ sigma @ weights):.4f}")

        # Create max return portfolio
        print("\n--- Maximum Return Portfolio (10% vol constraint) ---")
        cons = [LongOnlyConstraint(), FullInvestmentConstraint(),
                VolatilityConstraint(tickers, sigma, 0.10)]
        max_ret = MaxExpectedReturnOptimizer(tickers, cons, expected_returns)
        max_ret.solve()
        weights = max_ret.get_weights()
        print("Optimal Weights:")
        print(weights.round(4))
    else:
        print("\n(cvxpy not available - showing sample equal weights)")
        print(pd.Series(1/len(tickers), tickers))


def demo_hierarchical_risk_parity():
    """Demonstrate Hierarchical Risk Parity."""
    print("\n" + "="*60)
    print("HIERARCHICAL RISK PARITY DEMO")
    print("="*60)

    # Sample data
    tickers = ['VTI', 'VEA', 'VWO', 'AGG', 'BNDX', 'EMB']

    # Sample covariance and correlation
    cov = np.array([
        [0.0287, 0.0250, 0.0267, 0.0000, 0.0002, 0.0084],
        [0.0250, 0.0281, 0.0288, 0.0003, 0.0002, 0.0092],
        [0.0267, 0.0288, 0.0414, 0.0005, 0.0004, 0.0112],
        [0.0000, 0.0003, 0.0005, 0.0017, 0.0008, 0.0019],
        [0.0002, 0.0002, 0.0004, 0.0008, 0.0010, 0.0011],
        [0.0084, 0.0092, 0.0112, 0.0019, 0.0011, 0.0083]
    ])
    cov = pd.DataFrame(cov, tickers, tickers)

    # Calculate correlation from covariance
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)

    print("\nCovariance Matrix:")
    print(cov)

    print("\nCorrelation Matrix:")
    print(corr.round(3))

    # Calculate HRP weights
    hrp_weights = hierarchical_risk_parity(cov, corr)

    print("\n--- HRP Portfolio Weights ---")
    print(hrp_weights.round(4))
    print(f"\nSum of weights: {hrp_weights.sum():.4f}")


def demo_portfolio_backtest():
    """Demonstrate portfolio backtesting."""
    print("\n" + "="*60)
    print("PORTFOLIO BACKTEST DEMO")
    print("="*60)

    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    tickers = ['VTI', 'AGG']

    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        returns = np.random.randn(len(dates)) * 0.015 + 0.0003
        prices[ticker] = 100 * np.exp(np.cumsum(returns))

    prices.index = pd.Index(map(lambda x: x.date(), prices.index))

    print(f"\nBacktest Period: {prices.index[0]} to {prices.index[-1]}")
    print(f"Assets: {tickers}")

    # Target weights
    target_weights = pd.Series([0.6, 0.4], tickers)

    # Create rebalancer
    rebalancer = ThresholdRebalancer(target_weights, threshold=0.05)

    # Run backtest
    backtest = Backtest(prices, cash_buffer=0.02)
    results = backtest.run(
        target_weights=target_weights,
        start_date='2020-01-01',
        end_date='2023-12-31',
        starting_investment=100000,
        rebalancer=rebalancer
    )

    print("\n--- Backtest Results ---")
    print(f"Starting Value: $100,000")
    print(f"Ending Value: ${results['portfolio_value'].iloc[-1]:,.2f}")

    total_return = (results['portfolio_value'].iloc[-1] / 100000) - 1
    years = (results.index[-1] - results.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1/years) - 1

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility: {results['portfolio_value'].pct_change().std() * np.sqrt(252):.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    results['portfolio_value'].plot(ax=ax)
    ax.set_title('Portfolio Value Over Time')
    ax.set_ylabel('Value ($)')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('/Users/tux/claudestuff/robo-advisor/backtest_results.png')
    print("\nChart saved to: backtest_results.png")
    plt.close()


def demo_tax_loss_harvesting():
    """Demonstrate Tax Loss Harvesting."""
    print("\n" + "="*60)
    print("TAX LOSS HARVESTING DEMO")
    print("="*60)

    # Sample tax lots
    lots_data = [
        {'ticker': 'VTI', 'purchase_price': 220, 'quantity': 100,
         'purchase_date': '2023-01-15', 'sell_price': np.nan, 'sell_date': np.nan},
        {'ticker': 'VTI', 'purchase_price': 210, 'quantity': 50,
         'purchase_date': '2023-06-15', 'sell_price': np.nan, 'sell_date': np.nan},
        {'ticker': 'VEA', 'purchase_price': 45, 'quantity': 200,
         'purchase_date': '2023-03-01', 'sell_price': np.nan, 'sell_date': np.nan},
    ]
    lots = pd.DataFrame(lots_data)

    current_prices = pd.Series({'VTI': 200, 'VEA': 48})
    current_date = dt.date(2024, 1, 15)

    print("\nCurrent Holdings:")
    print(lots)
    print(f"\nCurrent Prices: {current_prices.to_dict()}")
    print(f"Current Date: {current_date}")

    # Identify TLH opportunities
    etf_sets = [['VTI', 'SCHB', 'ITOT'], ['VEA', 'SCHF', 'VEU']]
    harvester = TaxLossHarvester(etf_sets)

    opportunities = harvester.identify_harvest_opportunities(lots, current_prices, current_date)

    print("\n--- Tax Loss Harvesting Opportunities ---")
    if len(opportunities) > 0:
        print(opportunities[['ticker', 'purchase_price', 'current_price', 'quantity', 'loss']])
    else:
        print("No harvestable losses found.")


def demo_retirement_withdrawals():
    """Demonstrate retirement withdrawal strategies."""
    print("\n" + "="*60)
    print("RETIREMENT WITHDRAWAL STRATEGIES DEMO")
    print("="*60)

    # Setup
    tax_calc = TaxCalculator()

    returns = {'taxable': 0.03, 'IRA': 0.04, 'Roth': 0.04}
    starting = {'taxable': 2000000, 'IRA': 1000000, 'Roth': 0}
    spending = 120000

    planner = RetirementWithdrawalPlanner(tax_calc, returns, starting, spending)

    print(f"\nStarting Values:")
    for k, v in starting.items():
        print(f"  {k}: ${v:,.0f}")
    print(f"Annual Spending: ${spending:,.0f}")

    # Test different strategies
    strategies = [
        ('IRA First', ['IRA', 'taxable', 'Roth']),
        ('Taxable First', ['taxable', 'IRA', 'Roth']),
    ]

    print("\n--- Sustainability Analysis ---")
    for name, order in strategies:
        years = planner.calculate_sustainability(order, max_years=50)
        print(f"{name}: {years:.1f} years")

    # Show detailed simulation for taxable-first strategy
    print("\n--- Detailed Simulation (Taxable First) ---")
    df = planner.simulate_withdrawals(['taxable', 'IRA', 'Roth'], years=10)
    print(df.round(0).astype(int))


def demo_complete_workflow():
    """Demonstrate complete robo-advisor workflow."""
    print("\n" + "="*60)
    print("COMPLETE ROBO-ADVISOR WORKFLOW")
    print("="*60)

def demo_monte_carlo_simulations():
    """Demonstrate Monte Carlo Simulations."""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATIONS DEMO")
    print("="*60)

    # Sample portfolio
    tickers = ['VTI', 'VEA', 'VWO', 'AGG']
    weights = pd.Series([0.5, 0.2, 0.1, 0.2], tickers)

    # Sample expected returns and covariance (annualized)
    expected_returns = pd.Series([0.07, 0.06, 0.08, 0.03], tickers)
    covariance = pd.DataFrame([
        [0.0289, 0.018, 0.022, 0.001],
        [0.018, 0.0256, 0.024, 0.002],
        [0.022, 0.024, 0.0400, 0.003],
        [0.001, 0.002, 0.003, 0.0009]
    ], tickers, tickers)

    print("\nPortfolio Composition:")
    for t, w in weights.items():
        print(f"  {t}: {w:.1%}")

    portfolio_return = (weights * expected_returns).sum()
    portfolio_vol = np.sqrt(weights @ covariance @ weights)
    print(f"\nPortfolio Expected Return: {portfolio_return:.2%}")
    print(f"Portfolio Volatility: {portfolio_vol:.2%}")

    # Create simulator
    mc = MonteCarloSimulator(expected_returns, covariance, weights, seed=42)

    # Simulation 1: Portfolio Value Projection
    print("\n--- 1. Portfolio Value Projection ---")
    print("Simulating 30-year portfolio growth...")

    initial_value = 100000
    years = 30
    n_sims = 1000

    value_paths = mc.simulate_portfolio_values(
        initial_value, years, n_sims, annual_contribution=5000
    )

    final_values = value_paths.iloc[-1]
    print(f"\nAfter {years} years:")
    print(f"  Median portfolio value: ${final_values.median():,.0f}")
    print(f"  Mean portfolio value: ${final_values.mean():,.0f}")
    print(f"  10th percentile: ${final_values.quantile(0.10):,.0f}")
    print(f"  90th percentile: ${final_values.quantile(0.90):,.0f}")
    print(f"  Probability of gain: {(final_values > initial_value).mean():.1%}")

    # Simulation 2: Value at Risk (VaR)
    print("\n--- 2. Value at Risk (VaR) Analysis ---")
    print("Calculating potential losses at 95% confidence...")

    var_results = mc.calculate_var_cvar(initial_value, confidence_level=0.95,
                                        time_horizon_days=252, n_simulations=10000)

    print(f"\n95% Value at Risk (1 year):")
    print(f"  VaR: ${var_results['VaR']:,.0f} ({var_results['VaR_pct']:.1%})")
    print(f"  CVaR (Expected Shortfall): ${var_results['CVaR']:,.0f} ({var_results['CVaR_pct']:.1%})")
    print(f"\nInterpretation: There is a 5% chance of losing more than")
    print(f"${var_results['VaR']:,.0f} over the next year.")

    # Simulation 3: Retirement Sustainability
    print("\n--- 3. Retirement Sustainability Analysis ---")
    print("Simulating 30-year retirement with $120k annual withdrawals...")

    starting_balance = {'taxable': 2000000, 'IRA': 1000000, 'Roth': 0}
    annual_withdrawal = 120000

    retirement_sims = mc.simulate_retirement_sustainability(
        starting_balance, annual_withdrawal, years=30, n_simulations=1000
    )

    success_rate = retirement_sims['Success_Rate'].iloc[0]
    print(f"\nRetirement Sustainability:")
    print(f"  Success Rate (30 years): {success_rate:.1%}")
    print(f"  Median final balance: ${retirement_sims['P50'].iloc[-1]:,.0f}")
    print(f"  10th percentile final: ${retirement_sims['P10'].iloc[-1]:,.0f}")
    print(f"  90th percentile final: ${retirement_sims['P90'].iloc[-1]:,.0f}")

    print("\n" + "="*60)


def demo_complete_workflow():

    # Step 1: Determine asset allocation using HRP
    print("\n--- Step 1: Asset Allocation (HRP) ---")
    tickers = ['VTI', 'VEA', 'VWO', 'AGG']

    # Sample covariance
    np.random.seed(42)
    returns_sample = np.random.multivariate_normal(
        [0.0005, 0.0004, 0.0006, 0.0002],
        [[0.0002, 0.00015, 0.00018, 0.00002],
         [0.00015, 0.00025, 0.00022, 0.00003],
         [0.00018, 0.00022, 0.00035, 0.00002],
         [0.00002, 0.00003, 0.00002, 0.00005]], 252
    )

    cov = pd.DataFrame(np.cov(returns_sample.T), tickers, tickers)
    corr = pd.DataFrame(np.corrcoef(returns_sample.T), tickers, tickers)

    hrp_weights = hierarchical_risk_parity(cov, corr)
    print("HRP Allocation:")
    for t, w in hrp_weights.items():
        print(f"  {t}: {w:.1%}")

    # Step 2: Show sample portfolio
    print("\n--- Step 2: Sample Portfolio ($100,000) ---")
    portfolio_value = 100000
    for t, w in hrp_weights.items():
        allocation = portfolio_value * w
        print(f"  {t}: ${allocation:,.0f} ({w:.1%})")

    print("\n--- Step 3: Ongoing Management ---")
    print("Rebalancing triggers when any asset deviates > 5% from target")
    print("Tax loss harvesting monitors for losses > $1,000")

    print("\n--- Step 4: Withdrawal Phase ---")
    print("Tax-efficient withdrawal order: Taxable → IRA → Roth")
    print("This maximizes tax-deferred growth and minimizes taxes")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ROBO-ADVISOR WITH PYTHON - COMPLETE SYSTEM")
    print("="*60)

    # Run all demonstrations
    try:
        demo_mean_variance_optimization()
    except Exception as e:
        print(f"MVO Demo Error: {e}")

    try:
        demo_hierarchical_risk_parity()
    except Exception as e:
        print(f"HRP Demo Error: {e}")

    try:
        demo_tax_loss_harvesting()
    except Exception as e:
        print(f"TLH Demo Error: {e}")

    try:
        demo_portfolio_backtest()
    except Exception as e:
        print(f"Backtest Demo Error: {e}")

    try:
        demo_retirement_withdrawals()
    except Exception as e:
        print(f"Withdrawal Demo Error: {e}")

    try:
        demo_complete_workflow()
    except Exception as e:
        print(f"Workflow Demo Error: {e}")

    try:
        demo_monte_carlo_simulations()
    except Exception as e:
        print(f"Monte Carlo Demo Error: {e}")

    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*60)
    print("""
To use this robo-advisor for your own portfolio:

1. Install dependencies:
   pip install numpy pandas matplotlib yfinance cvxpy scipy

2. Import the module:
   from robo_advisor import *

3. Define your assets and run optimization:
   tickers = ['VTI', 'VEA', 'VWO', 'AGG']
   prices = get_prices(tickers, '2020-01-01', '2024-01-01')

4. Calculate optimal weights using HRP:
   returns = prices.pct_change().dropna()
   cov = returns.cov() * 252
   corr = returns.corr()
   weights = hierarchical_risk_parity(cov, corr)

5. Run backtest:
   rebalancer = ThresholdRebalancer(weights, threshold=0.05)
   backtest = Backtest(prices)
   results = backtest.run(weights, '2020-01-01', '2024-01-01', 100000, rebalancer)

See the individual demo functions for more examples!
    """)
