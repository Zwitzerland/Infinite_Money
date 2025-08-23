"""
Performance metrics calculation for Infinite_Money trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats


class PerformanceMetrics:
    """Calculate various performance metrics for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance metrics calculator."""
        self.risk_free_rate = risk_free_rate
        self.annual_factor = 252  # Trading days per year
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sharpe ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / self.annual_factor)
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float((excess_returns.mean() * self.annual_factor) / (excess_returns.std() * np.sqrt(self.annual_factor)))
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sortino ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / self.annual_factor)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return float((excess_returns.mean() * self.annual_factor) / (downside_returns.std() * np.sqrt(self.annual_factor)))
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = self.calculate_max_drawdown(cumulative_returns)
        
        if max_drawdown == 0:
            return 0.0
        
        annual_return = (cumulative_returns.iloc[-1] - 1) * (self.annual_factor / len(returns))
        return float(annual_return / max_drawdown)
    
    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(values) == 0:
            return 0.0
        
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return float(abs(drawdown.min()))
    
    def calculate_var(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, alpha * 100))
    
    def calculate_cvar(self, returns: pd.Series, alpha: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, alpha)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return float(tail_returns.mean())
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        """Calculate information ratio."""
        if benchmark_returns is None:
            # Use zero as benchmark
            active_returns = returns
        else:
            # Align returns and benchmark
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) == 0:
                return 0.0
            
            active_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        
        if len(active_returns) == 0 or active_returns.std() == 0:
            return 0.0
        
        return float((active_returns.mean() * self.annual_factor) / (active_returns.std() * np.sqrt(self.annual_factor)))
    
    def calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> float:
        """Calculate Treynor ratio."""
        if beta == 0:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / self.annual_factor)
        return float((excess_returns.mean() * self.annual_factor) / beta)
    
    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)
    
    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) > 0)
        gross_loss = abs(sum(trade.get("pnl", 0) for trade in trades if trade.get("pnl", 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        return float(returns.std() * np.sqrt(self.annual_factor))
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 0.0
        
        covariance = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1]
        market_variance = np.var(aligned_data.iloc[:, 1])
        
        return float(covariance / market_variance) if market_variance > 0 else 0.0
    
    def calculate_alpha(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate alpha relative to market."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        beta = self.calculate_beta(returns, market_returns)
        excess_return = returns.mean() - (self.risk_free_rate / self.annual_factor)
        market_excess_return = market_returns.mean() - (self.risk_free_rate / self.annual_factor)
        
        return float(excess_return - beta * market_excess_return)
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate return skewness."""
        if len(returns) == 0:
            return 0.0
        
        return float(stats.skew(returns))
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate return kurtosis."""
        if len(returns) == 0:
            return 0.0
        
        return float(stats.kurtosis(returns))
    
    def calculate_jensen_alpha(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Jensen's alpha."""
        return self.calculate_alpha(returns, market_returns)
    
    def calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) == 0:
            return 0.0
        
        active_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        return float(active_returns.std() * np.sqrt(self.annual_factor))
    
    def calculate_ulcer_index(self, values: pd.Series) -> float:
        """Calculate Ulcer Index."""
        if len(values) == 0:
            return 0.0
        
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        squared_drawdown = drawdown ** 2
        
        return float(np.sqrt(squared_drawdown.mean()))
    
    def calculate_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Pain Ratio."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        ulcer_index = self.calculate_ulcer_index(cumulative_returns)
        
        if ulcer_index == 0:
            return 0.0
        
        total_return = cumulative_returns.iloc[-1] - 1
        return float(total_return / ulcer_index)
    
    def calculate_all_metrics(self, returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Calculate all performance metrics."""
        metrics = {}
        
        # Basic return metrics
        if len(returns) > 0:
            metrics["total_return"] = float((1 + returns).prod() - 1)
            metrics["annualized_return"] = float(((1 + returns).prod()) ** (self.annual_factor / len(returns)) - 1)
            metrics["volatility"] = self.calculate_volatility(returns)
            metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
            metrics["sortino_ratio"] = self.calculate_sortino_ratio(returns)
            metrics["calmar_ratio"] = self.calculate_calmar_ratio(returns)
            metrics["max_drawdown"] = self.calculate_max_drawdown((1 + returns).cumprod())
            metrics["var_95"] = self.calculate_var(returns, 0.05)
            metrics["cvar_95"] = self.calculate_cvar(returns, 0.05)
            metrics["skewness"] = self.calculate_skewness(returns)
            metrics["kurtosis"] = self.calculate_kurtosis(returns)
            metrics["ulcer_index"] = self.calculate_ulcer_index((1 + returns).cumprod())
            metrics["pain_ratio"] = self.calculate_pain_ratio(returns)
        
        # Benchmark-relative metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            metrics["information_ratio"] = self.calculate_information_ratio(returns, benchmark_returns)
            metrics["tracking_error"] = self.calculate_tracking_error(returns, benchmark_returns)
            metrics["beta"] = self.calculate_beta(returns, benchmark_returns)
            metrics["alpha"] = self.calculate_alpha(returns, benchmark_returns)
            metrics["jensen_alpha"] = self.calculate_jensen_alpha(returns, benchmark_returns)
        
        # Trade-based metrics
        if trades:
            metrics["win_rate"] = self.calculate_win_rate(trades)
            metrics["profit_factor"] = self.calculate_profit_factor(trades)
        
        return metrics
    
    def generate_performance_report(self, returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.calculate_all_metrics(returns, benchmark_returns, trades)
        
        report = {
            "summary": {
                "total_return": metrics.get("total_return", 0),
                "annualized_return": metrics.get("annualized_return", 0),
                "volatility": metrics.get("volatility", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
            },
            "risk_metrics": {
                "var_95": metrics.get("var_95", 0),
                "cvar_95": metrics.get("cvar_95", 0),
                "ulcer_index": metrics.get("ulcer_index", 0),
                "pain_ratio": metrics.get("pain_ratio", 0),
            },
            "risk_adjusted_returns": {
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0),
                "information_ratio": metrics.get("information_ratio", 0),
            },
            "distribution_metrics": {
                "skewness": metrics.get("skewness", 0),
                "kurtosis": metrics.get("kurtosis", 0),
            }
        }
        
        if benchmark_returns is not None:
            report["benchmark_metrics"] = {
                "beta": metrics.get("beta", 0),
                "alpha": metrics.get("alpha", 0),
                "tracking_error": metrics.get("tracking_error", 0),
            }
        
        if trades:
            report["trade_metrics"] = {
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "total_trades": len(trades),
            }
        
        return report