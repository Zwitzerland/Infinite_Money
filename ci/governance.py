"""
v6 Governance System - Non-negotiable Trade Gates
Every trade must clear: (i) DRO-Kelly feasible, (ii) CDaR/L-VaR within budget, (iii) MOT-admissible.
Fail any ⇒ flatten.

FTAP sanity stays permanent—this repo will never promise riskless, levered doubling.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from ..optimizer.dro_kelly.solver import WassersteinDROKellyOptimizer, DROKellyConfig
from ..optimizer.risks.cdar import CDaRCalculator, CDaRConfig, CDaRPathController
from ..optimizer.risks.lvar import LVaRCalculator, LVaRConfig, LVaRRiskController
from ..risk.martingale_optimal_transport import MartingaleOptimalTransport, MOTConfig
from ..utils.logger import get_logger


@dataclass
class GovernanceConfig:
    """Configuration for governance system."""
    # DRO Kelly parameters
    wasserstein_radius: float = 0.1
    kelly_leverage_cap: float = 2.0
    
    # CDaR parameters
    cdar_budget: float = 0.05  # 5% maximum CDaR
    cdar_confidence: float = 0.95
    
    # L-VaR parameters
    lvar_spread_threshold: float = 0.005  # 50bps
    lvar_confidence: float = 0.95
    
    # MOT parameters
    mot_confidence: float = 0.95
    mot_scenarios: int = 1000
    
    # Emergency settings
    emergency_flatten_threshold: float = 3  # Number of failed checks
    cooldown_period: int = 300  # 5 minutes cooldown after flatten


@dataclass
class TradeRequest:
    """Structured trade request."""
    proposed_weights: np.ndarray
    current_weights: np.ndarray
    returns_data: np.ndarray
    liquidity_data: Dict[str, Any]
    market_conditions: Dict[str, Any]
    timestamp: datetime
    request_id: str


@dataclass 
class GovernanceDecision:
    """Governance decision result."""
    approved: bool
    decision_id: str
    timestamp: datetime
    
    # Check results
    dro_kelly_feasible: bool
    cdar_satisfied: bool
    lvar_satisfied: bool
    mot_admissible: bool
    
    # Detailed diagnostics
    dro_kelly_diagnostics: Dict[str, Any]
    cdar_diagnostics: Dict[str, Any]
    lvar_diagnostics: Dict[str, Any]
    mot_diagnostics: Dict[str, Any]
    
    # Actions
    required_actions: List[str]
    emergency_flatten: bool
    cooldown_until: Optional[datetime]


class TradeGovernor:
    """
    v6 Trade Governor - The Final Gatekeeper
    
    Implements the three non-negotiable checks:
    1. DRO-Kelly feasibility
    2. CDaR/L-VaR budget compliance  
    3. MOT superhedging admissibility
    
    Mathematical reality: No infinite money, no riskless leverage.
    """
    
    def __init__(self, config: GovernanceConfig):
        """Initialize the trade governor."""
        self.config = config
        self.logger = get_logger("TradeGovernor")
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.decision_history = []
        self.failed_checks_count = 0
        self.last_flatten_time = None
        self.in_cooldown = False
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_reason = ""
        
        self.logger.info("Trade Governor v6 initialized - Mathematical reality enforced")
    
    def _initialize_components(self):
        """Initialize all governance components."""
        # DRO Kelly optimizer
        dro_config = DROKellyConfig(
            wasserstein_radius=self.config.wasserstein_radius,
            leverage_cap=self.config.kelly_leverage_cap
        )
        self.dro_kelly = WassersteinDROKellyOptimizer(dro_config)
        
        # CDaR calculator and controller
        cdar_config = CDaRConfig(
            confidence_level=self.config.cdar_confidence,
            budget=self.config.cdar_budget
        )
        self.cdar_calculator = CDaRCalculator(cdar_config)
        self.cdar_controller = CDaRPathController(cdar_config)
        
        # L-VaR calculator and controller
        lvar_config = LVaRConfig(
            confidence_level=self.config.lvar_confidence,
            bid_ask_spread_threshold=self.config.lvar_spread_threshold
        )
        self.lvar_calculator = LVaRCalculator(lvar_config)
        self.lvar_controller = LVaRRiskController(lvar_config)
        
        # MOT superhedging
        mot_config = MOTConfig(
            confidence_level=self.config.mot_confidence,
            num_scenarios=self.config.mot_scenarios
        )
        self.mot_superhedging = MartingaleOptimalTransport(mot_config)
    
    def evaluate_trade_request(self, trade_request: TradeRequest) -> GovernanceDecision:
        """
        Evaluate trade request against all governance checks.
        
        This is the critical gate - every trade must pass ALL checks.
        """
        decision_id = f"GOV_{trade_request.request_id}_{int(datetime.now().timestamp())}"
        
        self.logger.info(f"Evaluating trade request {trade_request.request_id}")
        
        # Check cooldown period
        if self.in_cooldown:
            return self._create_cooldown_decision(decision_id, trade_request)
        
        # Emergency mode check
        if self.emergency_mode:
            return self._create_emergency_decision(decision_id, trade_request)
        
        # Execute the three non-negotiable checks
        try:
            # Check 1: DRO-Kelly Feasibility
            dro_kelly_result = self._check_dro_kelly_feasibility(trade_request)
            
            # Check 2: CDaR/L-VaR Budget Compliance
            cdar_result = self._check_cdar_budget(trade_request)
            lvar_result = self._check_lvar_budget(trade_request)
            
            # Check 3: MOT Superhedging Admissibility
            mot_result = self._check_mot_admissibility(trade_request)
            
            # Aggregate results
            all_checks_passed = (
                dro_kelly_result["feasible"] and
                cdar_result["satisfied"] and  
                lvar_result["satisfied"] and
                mot_result["admissible"]
            )
            
            # Create decision
            decision = GovernanceDecision(
                approved=all_checks_passed,
                decision_id=decision_id,
                timestamp=datetime.now(),
                dro_kelly_feasible=dro_kelly_result["feasible"],
                cdar_satisfied=cdar_result["satisfied"],
                lvar_satisfied=lvar_result["satisfied"],
                mot_admissible=mot_result["admissible"],
                dro_kelly_diagnostics=dro_kelly_result,
                cdar_diagnostics=cdar_result,
                lvar_diagnostics=lvar_result,
                mot_diagnostics=mot_result,
                required_actions=[],
                emergency_flatten=False,
                cooldown_until=None
            )
            
            # Handle failed checks
            if not all_checks_passed:
                decision = self._handle_failed_checks(decision, trade_request)
            
            # Log decision
            self._log_decision(decision, trade_request)
            
            # Update state
            self.decision_history.append(decision)
            if not all_checks_passed:
                self.failed_checks_count += 1
            else:
                self.failed_checks_count = 0  # Reset on success
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in governance evaluation: {str(e)}")
            return self._create_error_decision(decision_id, trade_request, str(e))
    
    def _check_dro_kelly_feasibility(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check 1: DRO-Kelly feasibility."""
        try:
            # Optimize using DRO-Kelly
            optimal_weights, diagnostics = self.dro_kelly.optimize(
                trade_request.returns_data, 
                trade_request.market_conditions
            )
            
            # Check if proposed weights are feasible
            proposed_leverage = np.sum(np.abs(trade_request.proposed_weights))
            max_leverage = self.config.kelly_leverage_cap
            
            # Leverage feasibility
            leverage_feasible = proposed_leverage <= max_leverage
            
            # Growth rate comparison
            proposed_growth = self.dro_kelly._compute_worst_case_growth(
                trade_request.proposed_weights,
                trade_request.returns_data,
                self.config.wasserstein_radius
            )
            
            optimal_growth = diagnostics.get("worst_case_growth", 0)
            
            # Growth feasibility (proposed shouldn't be much worse than optimal)
            growth_tolerance = 0.02  # 2% tolerance
            growth_feasible = (proposed_growth >= optimal_growth - growth_tolerance)
            
            feasible = leverage_feasible and growth_feasible
            
            return {
                "feasible": feasible,
                "leverage_feasible": leverage_feasible,
                "growth_feasible": growth_feasible,
                "proposed_leverage": proposed_leverage,
                "max_leverage": max_leverage,
                "proposed_growth": proposed_growth,
                "optimal_growth": optimal_growth,
                "optimal_weights": optimal_weights,
                "diagnostics": diagnostics
            }
            
        except Exception as e:
            self.logger.error(f"Error in DRO-Kelly feasibility check: {str(e)}")
            return {"feasible": False, "error": str(e)}
    
    def _check_cdar_budget(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check 2a: CDaR budget compliance."""
        try:
            # Check CDaR constraint
            constraint_check = self.cdar_calculator.check_cdar_constraint(
                trade_request.returns_data,
                trade_request.proposed_weights
            )
            
            # Check breach risk with controller
            breach_analysis = self.cdar_controller.check_breach_risk(
                trade_request.proposed_weights
            )
            
            # Determine satisfaction
            satisfied = (
                constraint_check["overall_satisfied"] and
                breach_analysis.get("risk_level", "HIGH") in ["LOW", "MEDIUM"]
            )
            
            return {
                "satisfied": satisfied,
                "constraint_check": constraint_check,
                "breach_analysis": breach_analysis,
                "budget": self.config.cdar_budget
            }
            
        except Exception as e:
            self.logger.error(f"Error in CDaR budget check: {str(e)}")
            return {"satisfied": False, "error": str(e)}
    
    def _check_lvar_budget(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check 2b: L-VaR budget compliance."""
        try:
            # Compute L-VaR
            lvar_value, lvar_diagnostics = self.lvar_calculator.compute_lvar(
                trade_request.returns_data,
                trade_request.proposed_weights,
                trade_request.liquidity_data
            )
            
            # Check liquidity constraints
            liquidity_constraints = self.lvar_calculator.check_liquidity_constraints(
                trade_request.proposed_weights,
                trade_request.liquidity_data
            )
            
            # Get leverage cap
            leverage_cap, cap_diagnostics = self.lvar_controller.compute_leverage_cap(
                trade_request.proposed_weights,
                trade_request.liquidity_data
            )
            
            # Check satisfaction
            proposed_leverage = np.sum(np.abs(trade_request.proposed_weights))
            leverage_ok = proposed_leverage <= leverage_cap
            liquidity_ok = liquidity_constraints["overall_satisfied"]
            
            satisfied = leverage_ok and liquidity_ok and not liquidity_constraints["emergency_conditions"]
            
            return {
                "satisfied": satisfied,
                "lvar_value": lvar_value,
                "leverage_cap": leverage_cap,
                "proposed_leverage": proposed_leverage,
                "leverage_ok": leverage_ok,
                "liquidity_ok": liquidity_ok,
                "lvar_diagnostics": lvar_diagnostics,
                "liquidity_constraints": liquidity_constraints,
                "cap_diagnostics": cap_diagnostics
            }
            
        except Exception as e:
            self.logger.error(f"Error in L-VaR budget check: {str(e)}")
            return {"satisfied": False, "error": str(e)}
    
    def _check_mot_admissibility(self, trade_request: TradeRequest) -> Dict[str, Any]:
        """Check 3: MOT superhedging admissibility."""
        try:
            # Compute price bands
            price_band = self.mot_superhedging.compute_price_bands(
                trade_request.returns_data,
                self._simple_payoff_function,
                trade_request.market_conditions
            )
            
            # Compute hedge bands
            target_payoff = np.sum(trade_request.proposed_weights) * 0.1  # 10% of position
            hedge_band = self.mot_superhedging.compute_hedge_bands(
                trade_request.returns_data,
                target_payoff,
                trade_request.market_conditions
            )
            
            # Check admissibility
            admissibility = self.mot_superhedging.check_superhedge_admissibility(
                trade_request.proposed_weights,
                price_band,
                hedge_band
            )
            
            return {
                "admissible": admissibility["admissible"],
                "price_band": price_band,
                "hedge_band": hedge_band,
                "admissibility_details": admissibility
            }
            
        except Exception as e:
            self.logger.error(f"Error in MOT admissibility check: {str(e)}")
            return {"admissible": False, "error": str(e)}
    
    def _simple_payoff_function(self, prices: np.ndarray) -> float:
        """Simple payoff function for MOT."""
        if len(prices) == 0:
            return 0.0
        return prices[-1] - prices[0]
    
    def _handle_failed_checks(self, decision: GovernanceDecision, trade_request: TradeRequest) -> GovernanceDecision:
        """Handle failed governance checks."""
        actions = []
        
        # Analyze failures
        if not decision.dro_kelly_feasible:
            actions.append("REDUCE_LEVERAGE_DRO_KELLY")
            
        if not decision.cdar_satisfied:
            actions.append("REDUCE_POSITIONS_CDAR_BREACH")
            
        if not decision.lvar_satisfied:
            actions.append("REDUCE_ILLIQUID_POSITIONS")
            
        if not decision.mot_admissible:
            actions.append("ADJUST_TO_MOT_BOUNDS")
        
        # Check for emergency flatten condition
        if self.failed_checks_count >= self.config.emergency_flatten_threshold:
            actions.append("EMERGENCY_FLATTEN")
            decision.emergency_flatten = True
            decision.cooldown_until = datetime.now().timestamp() + self.config.cooldown_period
            
            self.logger.critical(f"EMERGENCY FLATTEN triggered after {self.failed_checks_count} failed checks")
            
            # Enter emergency mode
            self.emergency_mode = True
            self.emergency_reason = f"Multiple governance failures: {self.failed_checks_count}"
            self.last_flatten_time = datetime.now()
        
        decision.required_actions = actions
        
        return decision
    
    def _create_cooldown_decision(self, decision_id: str, trade_request: TradeRequest) -> GovernanceDecision:
        """Create decision during cooldown period."""
        return GovernanceDecision(
            approved=False,
            decision_id=decision_id,
            timestamp=datetime.now(),
            dro_kelly_feasible=False,
            cdar_satisfied=False,
            lvar_satisfied=False,
            mot_admissible=False,
            dro_kelly_diagnostics={"cooldown": True},
            cdar_diagnostics={"cooldown": True},
            lvar_diagnostics={"cooldown": True},
            mot_diagnostics={"cooldown": True},
            required_actions=["WAIT_COOLDOWN"],
            emergency_flatten=False,
            cooldown_until=self.last_flatten_time.timestamp() + self.config.cooldown_period if self.last_flatten_time else None
        )
    
    def _create_emergency_decision(self, decision_id: str, trade_request: TradeRequest) -> GovernanceDecision:
        """Create decision during emergency mode."""
        return GovernanceDecision(
            approved=False,
            decision_id=decision_id,
            timestamp=datetime.now(),
            dro_kelly_feasible=False,
            cdar_satisfied=False,
            lvar_satisfied=False,
            mot_admissible=False,
            dro_kelly_diagnostics={"emergency_mode": True},
            cdar_diagnostics={"emergency_mode": True},
            lvar_diagnostics={"emergency_mode": True},
            mot_diagnostics={"emergency_mode": True},
            required_actions=["EMERGENCY_MODE_ACTIVE"],
            emergency_flatten=True,
            cooldown_until=None
        )
    
    def _create_error_decision(self, decision_id: str, trade_request: TradeRequest, error_msg: str) -> GovernanceDecision:
        """Create decision for error cases."""
        return GovernanceDecision(
            approved=False,
            decision_id=decision_id,
            timestamp=datetime.now(),
            dro_kelly_feasible=False,
            cdar_satisfied=False,
            lvar_satisfied=False,
            mot_admissible=False,
            dro_kelly_diagnostics={"error": error_msg},
            cdar_diagnostics={"error": error_msg},
            lvar_diagnostics={"error": error_msg},
            mot_diagnostics={"error": error_msg},
            required_actions=["ERROR_FLATTEN"],
            emergency_flatten=True,
            cooldown_until=datetime.now().timestamp() + self.config.cooldown_period
        )
    
    def _log_decision(self, decision: GovernanceDecision, trade_request: TradeRequest):
        """Log governance decision."""
        if decision.approved:
            self.logger.info(f"TRADE APPROVED: {decision.decision_id}")
        else:
            failed_checks = []
            if not decision.dro_kelly_feasible:
                failed_checks.append("DRO-Kelly")
            if not decision.cdar_satisfied:
                failed_checks.append("CDaR")
            if not decision.lvar_satisfied:
                failed_checks.append("L-VaR")
            if not decision.mot_admissible:
                failed_checks.append("MOT")
            
            self.logger.warning(f"TRADE REJECTED: {decision.decision_id} - Failed: {', '.join(failed_checks)}")
            
            if decision.emergency_flatten:
                self.logger.critical(f"EMERGENCY FLATTEN TRIGGERED: {decision.decision_id}")
    
    def reset_emergency_mode(self, manual_override: bool = False):
        """Reset emergency mode (use with extreme caution)."""
        if manual_override:
            self.emergency_mode = False
            self.emergency_reason = ""
            self.failed_checks_count = 0
            self.in_cooldown = False
            self.logger.warning("Emergency mode manually reset - USE WITH EXTREME CAUTION")
        else:
            self.logger.error("Emergency mode reset requires manual override")
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get current governance system status."""
        return {
            "emergency_mode": self.emergency_mode,
            "emergency_reason": self.emergency_reason,
            "in_cooldown": self.in_cooldown,
            "failed_checks_count": self.failed_checks_count,
            "last_flatten_time": self.last_flatten_time.isoformat() if self.last_flatten_time else None,
            "total_decisions": len(self.decision_history),
            "approved_decisions": sum(1 for d in self.decision_history if d.approved),
            "rejection_rate": 1 - (sum(1 for d in self.decision_history if d.approved) / max(len(self.decision_history), 1))
        }


class FTAPSafetyCheck:
    """
    Fundamental Theorem of Asset Pricing (FTAP) Safety Check
    
    Permanent reminder: No free lunch, no riskless leverage, no infinite money.
    """
    
    @staticmethod
    def verify_no_arbitrage_claims(strategy_description: str) -> Tuple[bool, str]:
        """Verify that strategy doesn't claim arbitrage opportunities."""
        forbidden_terms = [
            "infinite money",
            "zero risk", 
            "guaranteed profit",
            "riskless",
            "arbitrage free money",
            "infinite returns",
            "no loss",
            "certain profit"
        ]
        
        description_lower = strategy_description.lower()
        
        for term in forbidden_terms:
            if term in description_lower:
                return False, f"FTAP VIOLATION: Strategy claims '{term}' - mathematically impossible"
        
        return True, "FTAP compliant - no impossible claims detected"
    
    @staticmethod
    def log_ftap_reminder():
        """Log permanent FTAP reminder."""
        logger = get_logger("FTAP_Safety")
        logger.info("FTAP REMINDER: In any arbitrage-free market, there is no strategy that guarantees positive returns without risk. All trading involves probabilistic outcomes.")
        logger.info("Mathematical reality: maximal geometric growth under hard loss constraints is the only achievable objective.")


# Initialize FTAP safety on module import
FTAPSafetyCheck.log_ftap_reminder()