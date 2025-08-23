"""
Configuration management for Infinite_Money trading system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """System-level configuration."""
    name: str = "Infinite_Money"
    version: str = "1.0.0"
    mode: str = "autonomous"
    quantum_enabled: bool = True
    gpu_acceleration: bool = True
    debug_mode: bool = False
    max_concurrent_tasks: int = 10
    memory_limit_gb: int = 32
    cpu_cores: int = 8
    log_level: str = "INFO"
    log_file: str = "logs/infinite_money.log"
    enable_structured_logging: bool = True


class AgentConfig(BaseModel):
    """Agent configuration."""
    enabled: bool = True
    update_frequency_minutes: int = 5
    strategy_evolution_rate: float = 0.1
    max_strategies_per_generation: int = 50


class QuantumConfig(BaseModel):
    """Quantum computing configuration."""
    backend: str = "qiskit"
    qubits: int = 32
    optimization_level: int = 2
    shots: int = 1000
    error_mitigation: bool = True


class TradingConfig(BaseModel):
    """Trading configuration."""
    markets: list = Field(default_factory=list)
    strategy_types: list = Field(default_factory=list)
    risk_management: Dict[str, Any] = Field(default_factory=dict)


class Config:
    """Main configuration class for Infinite_Money."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path or "configs/main_config.yaml"
        self._config_data = self._load_config()
        
        # Initialize sub-configs
        self.system = SystemConfig(**self._config_data.get("system", {}))
        self.agents = self._config_data.get("agents", {})
        self.quantum = QuantumConfig(**self._config_data.get("quantum", {}))
        self.trading = TradingConfig(**self._config_data.get("trading", {}))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name, {})
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """Check if an agent is enabled."""
        agent_config = self.get_agent_config(agent_name)
        return agent_config.get("enabled", True)
    
    def get_quantum_config(self) -> QuantumConfig:
        """Get quantum configuration."""
        return self.quantum
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return self.trading
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dict__'):
                    # Update nested config objects
                    current = getattr(self, key)
                    for k, v in value.items():
                        if hasattr(current, k):
                            setattr(current, k, v)
                else:
                    setattr(self, key, value)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        output_path = output_path or self.config_path
        
        config_data = {
            "system": self.system.dict(),
            "agents": self.agents,
            "quantum": self.quantum.dict(),
            "trading": self.trading.dict(),
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        # Basic validation
        if self.system.mode not in ["autonomous", "supervised", "backtest", "research"]:
            raise ValueError(f"Invalid system mode: {self.system.mode}")
        
        if self.quantum.backend not in ["qiskit", "cirq", "pennylane"]:
            raise ValueError(f"Invalid quantum backend: {self.quantum.backend}")
        
        if self.system.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        
        return True
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # System overrides
        if os.getenv("INFINITE_MONEY_MODE"):
            overrides["system"] = {"mode": os.getenv("INFINITE_MONEY_MODE")}
        
        if os.getenv("INFINITE_MONEY_DEBUG"):
            overrides["system"] = {"debug_mode": os.getenv("INFINITE_MONEY_DEBUG").lower() == "true"}
        
        # Quantum overrides
        if os.getenv("QUANTUM_BACKEND"):
            overrides["quantum"] = {"backend": os.getenv("QUANTUM_BACKEND")}
        
        return overrides
    
    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        overrides = self.get_env_overrides()
        self.update_config(overrides)