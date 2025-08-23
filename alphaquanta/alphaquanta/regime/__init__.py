"""
Regime detection modules for AlphaQuanta.
"""

from .hmm_detector import HiddenMarkovModel, MarketRegimeDetector

__all__ = [
    "HiddenMarkovModel",
    "MarketRegimeDetector",
]
