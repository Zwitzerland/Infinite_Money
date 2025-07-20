"""
ACU (Agent Compute Unit) tracking for budget management.
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ACUTracker:
    """Tracks ACU usage for budget compliance."""
    
    def __init__(self, budget: float = 20.0):
        self.budget = budget
        self.total_used = 0.0
        self.operations = {}
        self.operation_stack = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.operation_costs = {
            'process_trade_command': 0.1,
            'generate_signals': 0.2,
            'run_backtest': 1.0,
            'quantum_optimization': 0.5,
            'risk_validation': 0.05,
            'market_data_fetch': 0.02,
            'default': 0.1
        }
    
    def start_operation(self, operation_name: str) -> str:
        """Start tracking an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        start_time = time.time()
        self.operations[operation_id] = {
            'name': operation_name,
            'start_time': start_time,
            'end_time': None,
            'duration': None,
            'acu_cost': None
        }
        
        self.operation_stack.append(operation_id)
        
        self.logger.debug(f"Started operation: {operation_name} ({operation_id})")
        return operation_id
    
    def end_operation(self, operation_name: str) -> float:
        """End tracking an operation and calculate ACU cost."""
        if not self.operation_stack:
            self.logger.warning(f"No active operation to end: {operation_name}")
            return 0.0
        
        operation_id = None
        for op_id in reversed(self.operation_stack):
            if self.operations[op_id]['name'] == operation_name:
                operation_id = op_id
                break
        
        if not operation_id:
            self.logger.warning(f"Operation not found in stack: {operation_name}")
            return 0.0
        
        operation = self.operations[operation_id]
        end_time = time.time()
        duration = end_time - operation['start_time']
        
        base_cost = self.operation_costs.get(operation_name, self.operation_costs['default'])
        duration_multiplier = max(1.0, duration / 10.0)
        acu_cost = base_cost * duration_multiplier
        
        operation['end_time'] = end_time
        operation['duration'] = duration
        operation['acu_cost'] = acu_cost
        
        self.total_used += acu_cost
        self.operation_stack.remove(operation_id)
        
        self.logger.debug(f"Ended operation: {operation_name} - Duration: {duration:.2f}s, ACU: {acu_cost:.3f}")
        
        if self.total_used > self.budget:
            self.logger.warning(f"ACU budget exceeded: {self.total_used:.2f} > {self.budget}")
        
        return acu_cost
    
    def get_remaining_budget(self) -> float:
        """Get remaining ACU budget."""
        return max(0.0, self.budget - self.total_used)
    
    def get_usage_summary(self) -> Dict:
        """Get detailed usage summary."""
        completed_operations = [op for op in self.operations.values() if op['acu_cost'] is not None]
        
        by_operation = {}
        for op in completed_operations:
            name = op['name']
            if name not in by_operation:
                by_operation[name] = {'count': 0, 'total_acu': 0.0, 'total_duration': 0.0}
            
            by_operation[name]['count'] += 1
            by_operation[name]['total_acu'] += op['acu_cost']
            by_operation[name]['total_duration'] += op['duration']
        
        return {
            'total_budget': self.budget,
            'total_used': self.total_used,
            'remaining': self.get_remaining_budget(),
            'utilization_pct': (self.total_used / self.budget) * 100,
            'total_operations': len(completed_operations),
            'by_operation': by_operation,
            'active_operations': len(self.operation_stack)
        }
    
    def export_usage_log(self) -> str:
        """Export usage log as JSON."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_usage_summary(),
            'operations': list(self.operations.values())
        }
        
        return json.dumps(log_data, indent=2)
    
    def check_budget_compliance(self) -> bool:
        """Check if within budget limits."""
        return self.total_used <= self.budget
    
    def reset(self):
        """Reset tracker for new session."""
        self.total_used = 0.0
        self.operations.clear()
        self.operation_stack.clear()
        self.logger.info("ACU tracker reset")
