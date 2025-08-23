"""
Logging configuration for Infinite_Money trading system.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import structlog


class StructuredLogger:
    """Structured logging wrapper for Infinite_Money."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize structured logger."""
        self.config = config
        self.log_level = config.get("log_level", "INFO")
        self.log_file = config.get("log_file", "logs/infinite_money.log")
        self.enable_structured_logging = config.get("enable_structured_logging", True)
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup loguru logger with structured logging."""
        # Remove default handler
        logger.remove()
        
        # Create logs directory if it doesn't exist
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Console handler with structured logging
        if self.enable_structured_logging:
            logger.add(
                sys.stdout,
                format=self._structured_format,
                level=self.log_level,
                serialize=True
            )
        else:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=self.log_level
            )
        
        # File handler
        logger.add(
            self.log_file,
            format=self._structured_format if self.enable_structured_logging else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=self.log_level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            serialize=self.enable_structured_logging
        )
        
        # Error file handler
        error_log = str(log_path.parent / "errors.log")
        logger.add(
            error_log,
            format=self._structured_format if self.enable_structured_logging else "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="50 MB",
            retention="90 days",
            compression="zip",
            serialize=self.enable_structured_logging
        )
    
    def _structured_format(self, record: Dict[str, Any]) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "module": record["module"],
            "process": record["process"].id,
            "thread": record["thread"].id,
        }
        
        # Add extra fields if present
        if "extra" in record and record["extra"]:
            log_entry["extra"] = record["extra"]
        
        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        return json.dumps(log_entry)
    
    def get_logger(self, name: str = None):
        """Get logger instance."""
        return logger.bind(name=name) if name else logger
    
    def log_agent_action(self, agent_name: str, action: str, details: Dict[str, Any] = None):
        """Log agent action with structured data."""
        log_data = {
            "agent": agent_name,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            log_data.update(details)
        
        logger.info("Agent action", **log_data)
    
    def log_trading_event(self, event_type: str, symbol: str, details: Dict[str, Any] = None):
        """Log trading event with structured data."""
        log_data = {
            "event_type": event_type,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if details:
            log_data.update(details)
        
        logger.info("Trading event", **log_data)
    
    def log_quantum_operation(self, operation: str, circuit_info: Dict[str, Any] = None):
        """Log quantum operation with structured data."""
        log_data = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if circuit_info:
            log_data.update(circuit_info)
        
        logger.info("Quantum operation", **log_data)
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics with structured data."""
        log_data = {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Performance metrics", **log_data)
    
    def log_system_health(self, health_data: Dict[str, Any]):
        """Log system health metrics."""
        log_data = {
            "health": health_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("System health", **log_data)


def setup_logger(config: Dict[str, Any]) -> StructuredLogger:
    """Setup and return structured logger instance."""
    return StructuredLogger(config)


def get_logger(name: str = None):
    """Get logger instance for the given name."""
    return logger.bind(name=name) if name else logger


# Convenience functions for common logging patterns
def log_agent_start(agent_name: str, config: Dict[str, Any] = None):
    """Log agent startup."""
    logger.info(f"Starting agent: {agent_name}", agent=agent_name, config=config)


def log_agent_stop(agent_name: str, runtime_seconds: float = None):
    """Log agent shutdown."""
    logger.info(f"Stopping agent: {agent_name}", agent=agent_name, runtime_seconds=runtime_seconds)


def log_strategy_generation(strategy_name: str, performance: Dict[str, Any] = None):
    """Log strategy generation."""
    logger.info(f"Generated strategy: {strategy_name}", strategy=strategy_name, performance=performance)


def log_portfolio_update(portfolio_value: float, positions: Dict[str, Any] = None):
    """Log portfolio update."""
    logger.info("Portfolio updated", portfolio_value=portfolio_value, positions=positions)


def log_risk_alert(alert_type: str, details: Dict[str, Any]):
    """Log risk alert."""
    logger.warning(f"Risk alert: {alert_type}", alert_type=alert_type, **details)


def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """Log error with additional context."""
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if context:
        error_data.update(context)
    
    logger.error("Error occurred", **error_data, exc_info=True)