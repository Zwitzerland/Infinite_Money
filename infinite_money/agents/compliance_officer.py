"""
Compliance Officer Agent for Infinite_Money trading system.
"""

import asyncio
from typing import Dict, Any
from datetime import datetime

from .base_agent import BaseAgent, Task
from ..utils.logger import get_logger
from ..utils.config import Config


class ComplianceOfficerAgent(BaseAgent):
    """Compliance Officer Agent - Handles risk management and compliance."""
    
    def __init__(self, config: Config, agent_config: Dict[str, Any] = None):
        """Initialize Compliance Officer Agent."""
        super().__init__("ComplianceOfficer", config, agent_config)
        
        self.risk_limits = agent_config.get("risk_limits", {})
        self.trading_hours = agent_config.get("trading_hours", {})
        self.blacklist_symbols = agent_config.get("blacklist_symbols", [])
        self.whitelist_symbols = agent_config.get("whitelist_symbols", [])
        
        self.logger.info("Compliance Officer Agent initialized")
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute Compliance Officer tasks."""
        task_type = task.task_type
        data = task.data
        
        try:
            if task_type == "enforce_limits":
                return await self._enforce_limits(data)
            elif task_type == "monitor_risk":
                return await self._monitor_risk(data)
            elif task_type == "check_compliance":
                return await self._check_compliance(data)
            else:
                return {"status": "error", "message": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _enforce_limits(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce risk limits and trading restrictions."""
        portfolio_metrics = data.get("portfolio_metrics", {})
        risk_limit = data.get("risk_limit", 0.02)
        
        self.logger.info("Enforcing risk limits")
        
        try:
            violations = []
            
            # Check drawdown limit
            drawdown = portfolio_metrics.get("drawdown", 0)
            max_drawdown = self.risk_limits.get("max_drawdown", 0.15)
            if drawdown > max_drawdown:
                violations.append({
                    "type": "drawdown_violation",
                    "current": drawdown,
                    "limit": max_drawdown,
                    "severity": "high"
                })
            
            # Check VaR limit
            var_95 = portfolio_metrics.get("var_95", 0)
            max_var = self.risk_limits.get("var_95", 0.02)
            if var_95 > max_var:
                violations.append({
                    "type": "var_violation",
                    "current": var_95,
                    "limit": max_var,
                    "severity": "medium"
                })
            
            # Check concentration limit
            positions = portfolio_metrics.get("positions", {})
            for symbol, weight in positions.items():
                concentration_limit = self.risk_limits.get("concentration_limit", 0.25)
                if abs(weight) > concentration_limit:
                    violations.append({
                        "type": "concentration_violation",
                        "symbol": symbol,
                        "current": weight,
                        "limit": concentration_limit,
                        "severity": "medium"
                    })
            
            return {
                "status": "success",
                "violations": violations,
                "violations_count": len(violations),
                "risk_level": "high" if violations else "normal"
            }
            
        except Exception as e:
            self.logger.error(f"Error enforcing limits: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _monitor_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor portfolio risk metrics."""
        portfolio_metrics = data.get("portfolio_metrics", {})
        risk_limit = data.get("risk_limit", 0.02)
        
        self.logger.info("Monitoring portfolio risk")
        
        try:
            risk_metrics = {
                "total_value": portfolio_metrics.get("total_value", 0),
                "drawdown": portfolio_metrics.get("drawdown", 0),
                "var_95": portfolio_metrics.get("var_95", 0),
                "volatility": portfolio_metrics.get("volatility", 0),
                "beta": portfolio_metrics.get("beta", 1.0),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0)
            }
            
            # Calculate risk score
            risk_score = 0.0
            if risk_metrics["drawdown"] > 0.1:
                risk_score += 0.3
            if risk_metrics["var_95"] > 0.02:
                risk_score += 0.3
            if risk_metrics["volatility"] > 0.2:
                risk_score += 0.2
            if risk_metrics["sharpe_ratio"] < 0.5:
                risk_score += 0.2
            
            risk_level = "low" if risk_score < 0.3 else "medium" if risk_score < 0.7 else "high"
            
            return {
                "status": "success",
                "risk_metrics": risk_metrics,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "alerts": risk_score > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring risk: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _check_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check trading compliance rules."""
        orders = data.get("orders", [])
        current_time = datetime.now()
        
        self.logger.info("Checking trading compliance")
        
        try:
            compliance_issues = []
            
            # Check trading hours
            start_time = self.trading_hours.get("start", "09:30")
            end_time = self.trading_hours.get("end", "16:00")
            
            current_time_str = current_time.strftime("%H:%M")
            if not (start_time <= current_time_str <= end_time):
                compliance_issues.append({
                    "type": "trading_hours_violation",
                    "current_time": current_time_str,
                    "allowed_hours": f"{start_time}-{end_time}",
                    "severity": "medium"
                })
            
            # Check blacklist symbols
            for order in orders:
                symbol = order.get("symbol")
                if symbol in self.blacklist_symbols:
                    compliance_issues.append({
                        "type": "blacklist_violation",
                        "symbol": symbol,
                        "severity": "high"
                    })
            
            # Check whitelist (if specified)
            if self.whitelist_symbols:
                for order in orders:
                    symbol = order.get("symbol")
                    if symbol not in self.whitelist_symbols:
                        compliance_issues.append({
                            "type": "whitelist_violation",
                            "symbol": symbol,
                            "severity": "high"
                        })
            
            return {
                "status": "success",
                "compliance_issues": compliance_issues,
                "issues_count": len(compliance_issues),
                "compliant": len(compliance_issues) == 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking compliance: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent-specific metrics."""
        return {
            "risk_limits": self.risk_limits,
            "trading_hours": self.trading_hours,
            "blacklist_symbols": len(self.blacklist_symbols),
            "whitelist_symbols": len(self.whitelist_symbols)
        }