"""
QPU (Quantum Processing Unit) time tracking for quantum operations.
"""

import time
from typing import Dict, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class QPUTracker:
    """Tracks QPU time usage for quantum operations."""
    
    def __init__(self, budget_minutes: float = 10.0):
        self.budget = budget_minutes
        self.total_used = 0.0
        self.quantum_operations = {}
        self.active_operations = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.operation_estimates = {
            'qaoa_optimization': 2.0,
            'vqe_calculation': 1.5,
            'quantum_var': 0.8,
            'diffusion_forecast': 1.2,
            'amplitude_estimation': 0.5,
            'default': 1.0
        }
    
    def start_quantum_operation(self, operation_name: str, estimated_time: Optional[float] = None) -> str:
        """Start tracking a quantum operation."""
        operation_id = f"qpu_{operation_name}_{int(time.time() * 1000)}"
        
        if estimated_time is None:
            estimated_time = self.operation_estimates.get(operation_name, self.operation_estimates['default'])
        
        if self.total_used + estimated_time > self.budget:
            raise RuntimeError(f"QPU budget would be exceeded: {self.total_used + estimated_time:.2f} > {self.budget} minutes")
        
        start_time = time.time()
        self.active_operations[operation_id] = {
            'name': operation_name,
            'start_time': start_time,
            'estimated_time': estimated_time
        }
        
        self.logger.info(f"Started QPU operation: {operation_name} (estimated: {estimated_time:.2f} min)")
        return operation_id
    
    def end_quantum_operation(self, operation_id: str, actual_qpu_time: Optional[float] = None) -> float:
        """End tracking a quantum operation."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"QPU operation not found: {operation_id}")
            return 0.0
        
        operation = self.active_operations.pop(operation_id)
        end_time = time.time()
        wall_clock_duration = (end_time - operation['start_time']) / 60.0
        
        if actual_qpu_time is None:
            qpu_time_used = operation['estimated_time'] * 0.8
        else:
            qpu_time_used = actual_qpu_time
        
        self.quantum_operations[operation_id] = {
            'name': operation['name'],
            'start_time': operation['start_time'],
            'end_time': end_time,
            'wall_clock_duration': wall_clock_duration,
            'estimated_time': operation['estimated_time'],
            'actual_qpu_time': qpu_time_used,
            'efficiency': qpu_time_used / wall_clock_duration if wall_clock_duration > 0 else 0
        }
        
        self.total_used += qpu_time_used
        
        self.logger.info(f"Completed QPU operation: {operation['name']} - QPU time: {qpu_time_used:.2f} min")
        
        if self.total_used > self.budget:
            self.logger.error(f"QPU budget exceeded: {self.total_used:.2f} > {self.budget} minutes")
        
        return qpu_time_used
    
    def get_remaining_budget(self) -> float:
        """Get remaining QPU budget in minutes."""
        return max(0.0, self.budget - self.total_used)
    
    def get_usage_summary(self) -> Dict:
        """Get detailed QPU usage summary."""
        completed_operations = list(self.quantum_operations.values())
        
        by_operation = {}
        for op in completed_operations:
            name = op['name']
            if name not in by_operation:
                by_operation[name] = {
                    'count': 0, 
                    'total_qpu_time': 0.0, 
                    'total_wall_time': 0.0,
                    'avg_efficiency': 0.0
                }
            
            by_operation[name]['count'] += 1
            by_operation[name]['total_qpu_time'] += op['actual_qpu_time']
            by_operation[name]['total_wall_time'] += op['wall_clock_duration']
            by_operation[name]['avg_efficiency'] += op['efficiency']
        
        for name in by_operation:
            if by_operation[name]['count'] > 0:
                by_operation[name]['avg_efficiency'] /= by_operation[name]['count']
        
        return {
            'total_budget': self.budget,
            'total_used': self.total_used,
            'remaining': self.get_remaining_budget(),
            'utilization_pct': (self.total_used / self.budget) * 100,
            'total_operations': len(completed_operations),
            'active_operations': len(self.active_operations),
            'by_operation': by_operation,
            'avg_efficiency': sum(op['efficiency'] for op in completed_operations) / len(completed_operations) if completed_operations else 0
        }
    
    def export_usage_log(self) -> str:
        """Export QPU usage log as JSON."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_usage_summary(),
            'operations': list(self.quantum_operations.values()),
            'active_operations': list(self.active_operations.values())
        }
        
        return json.dumps(log_data, indent=2)
    
    def check_budget_compliance(self) -> bool:
        """Check if within QPU budget limits."""
        return self.total_used <= self.budget
    
    def get_governor_alert_threshold(self) -> float:
        """Get threshold for QPU governor alerts (80% of budget)."""
        return self.budget * 0.8
    
    def should_trigger_governor_alert(self) -> bool:
        """Check if QPU usage should trigger governor alert."""
        return self.total_used >= self.get_governor_alert_threshold()
    
    def reset(self):
        """Reset tracker for new session."""
        self.total_used = 0.0
        self.quantum_operations.clear()
        self.active_operations.clear()
        self.logger.info("QPU tracker reset")
