"""
PnL (Profit and Loss) monitoring for trading performance tracking.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class PnLMonitor:
    """Monitors profit and loss for trading performance."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.daily_pnl = {}
        self.positions = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'var_95': 0.0
        }
    
    def record_trade(self, trade_data: Dict) -> None:
        """Record a completed trade."""
        trade_record = {
            'trade_id': trade_data.get('trade_id'),
            'symbol': trade_data.get('symbol'),
            'side': trade_data.get('side', 'BUY'),
            'quantity': trade_data.get('quantity', 0),
            'execution_price': trade_data.get('execution_price', 0.0),
            'commission': trade_data.get('commission', 0.0),
            'timestamp': trade_data.get('execution_time', datetime.now().isoformat()),
            'strategy': trade_data.get('strategy', 'unknown'),
            'pnl': 0.0,
            'unrealized_pnl': 0.0
        }
        
        self.trades.append(trade_record)
        self._update_positions(trade_record)
        self._calculate_daily_pnl()
        
        self.logger.info(f"Recorded trade: {trade_record['symbol']} {trade_record['quantity']} @ {trade_record['execution_price']}")
    
    def _update_positions(self, trade: Dict) -> None:
        """Update position tracking."""
        symbol = trade['symbol']
        quantity = trade['quantity']
        price = trade['execution_price']
        side = trade['side']
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_cost': 0.0,
                'total_cost': 0.0,
                'unrealized_pnl': 0.0
            }
        
        position = self.positions[symbol]
        
        if side == 'BUY':
            new_total_cost = position['total_cost'] + (quantity * price)
            new_quantity = position['quantity'] + quantity
            position['avg_cost'] = new_total_cost / new_quantity if new_quantity > 0 else 0
            position['quantity'] = new_quantity
            position['total_cost'] = new_total_cost
        else:
            realized_pnl = quantity * (price - position['avg_cost'])
            self._record_realized_pnl(realized_pnl, trade)
            
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                position['quantity'] = 0
                position['avg_cost'] = 0.0
                position['total_cost'] = 0.0
    
    def _record_realized_pnl(self, pnl: float, trade: Dict) -> None:
        """Record realized PnL from trade."""
        trade['pnl'] = pnl
        self.current_capital += pnl
        
        date_key = datetime.fromisoformat(trade['timestamp']).date().isoformat()
        if date_key not in self.daily_pnl:
            self.daily_pnl[date_key] = 0.0
        self.daily_pnl[date_key] += pnl
    
    def _calculate_daily_pnl(self) -> None:
        """Calculate daily PnL including unrealized."""
        today = datetime.now().date().isoformat()
        
        realized_pnl = self.daily_pnl.get(today, 0.0)
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        self.daily_pnl[today] = realized_pnl + unrealized_pnl
    
    def update_market_prices(self, price_data: Dict[str, float]) -> None:
        """Update positions with current market prices."""
        for symbol, current_price in price_data.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                if position['quantity'] > 0:
                    position['unrealized_pnl'] = position['quantity'] * (current_price - position['avg_cost'])
        
        self._calculate_daily_pnl()
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return self.performance_metrics
        
        returns = self._calculate_returns()
        
        self.performance_metrics['total_return'] = (self.current_capital - self.initial_capital) / self.initial_capital
        self.performance_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
        self.performance_metrics['win_rate'] = self._calculate_win_rate()
        self.performance_metrics['profit_factor'] = self._calculate_profit_factor()
        self.performance_metrics['var_95'] = self._calculate_var(returns, 0.95)
        
        return self.performance_metrics
    
    def _calculate_returns(self) -> List[float]:
        """Calculate daily returns."""
        if len(self.daily_pnl) < 2:
            return [0.0]
        
        sorted_dates = sorted(self.daily_pnl.keys())
        returns = []
        
        for i in range(1, len(sorted_dates)):
            prev_capital = self.initial_capital + sum(self.daily_pnl[d] for d in sorted_dates[:i])
            current_pnl = self.daily_pnl[sorted_dates[i]]
            
            if prev_capital > 0:
                daily_return = current_pnl / prev_capital
                returns.append(daily_return)
        
        return returns if returns else [0.0]
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        risk_free_rate = 0.02 / 252
        excess_returns = returns_array - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown))
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage."""
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades = len([t for t in self.trades if t.get('pnl') is not None])
        
        return len(profitable_trades) / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        """Calculate Value at Risk."""
        if not returns or len(returns) < 10:
            return 0.0
        
        returns_array = np.array(returns)
        return float(np.percentile(returns_array, (1 - confidence) * 100))
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current position summary."""
        return {symbol: pos.copy() for symbol, pos in self.positions.items() if pos['quantity'] > 0}
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        metrics = self.calculate_performance_metrics()
        
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_return_pct': metrics['total_return'] * 100,
                'unrealized_pnl': sum(pos['unrealized_pnl'] for pos in self.positions.values())
            },
            'performance': {
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown'] * 100,
                'win_rate_pct': metrics['win_rate'] * 100,
                'profit_factor': metrics['profit_factor'],
                'var_95_pct': metrics['var_95'] * 100
            },
            'trading': {
                'total_trades': len(self.trades),
                'active_positions': len([p for p in self.positions.values() if p['quantity'] > 0]),
                'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in self.trades if t.get('pnl') is not None]) if self.trades else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def export_performance_log(self) -> str:
        """Export performance data as JSON."""
        log_data = {
            'summary': self.get_performance_summary(),
            'daily_pnl': self.daily_pnl,
            'positions': self.get_current_positions(),
            'recent_trades': self.trades[-10:] if len(self.trades) > 10 else self.trades
        }
        
        return json.dumps(log_data, indent=2)
    
    def reset(self, new_initial_capital: Optional[float] = None):
        """Reset monitor for new session."""
        if new_initial_capital:
            self.initial_capital = new_initial_capital
        
        self.current_capital = self.initial_capital
        self.trades.clear()
        self.daily_pnl.clear()
        self.positions.clear()
        
        for key in self.performance_metrics:
            self.performance_metrics[key] = 0.0
        
        self.logger.info("PnL monitor reset")
    
    def add_hmm_metrics(self, regime_detector) -> Dict[str, Any]:
        """Add HMM regime metrics to performance summary."""
        if not regime_detector or not regime_detector.current_regime:
            return {}
        
        hmm_metrics = {
            'hmm_current_regime': regime_detector.current_regime['current_regime'],
            'hmm_current_state': regime_detector.current_regime['current_state'],
            'hmm_confidence': regime_detector.current_regime['confidence'],
            'hmm_stability': regime_detector.current_regime['stability'],
            'hmm_state_distribution': regime_detector.current_regime['state_probabilities']
        }
        
        return hmm_metrics

    def export_executive_digest(self, regime_detector=None) -> str:
        """Export 60-minute executive digest with HMM state distribution."""
        summary = self.get_performance_summary()
        
        if regime_detector:
            summary['hmm_metrics'] = self.add_hmm_metrics(regime_detector)
        
        executive_digest = {
            'timestamp': datetime.now().isoformat(),
            'pnl_summary': {
                'current_capital': summary['capital']['current'],
                'total_return_pct': summary['capital']['total_return_pct'],
                'sharpe_ratio': summary['performance']['sharpe_ratio']
            },
            'risk_metrics': {
                'var_95_pct': summary['performance']['var_95_pct'],
                'max_drawdown_pct': summary['performance']['max_drawdown_pct']
            },
            'hmm_state_distribution': summary.get('hmm_metrics', {}).get('hmm_state_distribution', []),
            'regime_info': {
                'current_regime': summary.get('hmm_metrics', {}).get('hmm_current_regime', 'unknown'),
                'confidence': summary.get('hmm_metrics', {}).get('hmm_confidence', 0.0)
            }
        }
        
        return json.dumps(executive_digest, indent=2)
