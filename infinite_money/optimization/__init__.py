"""
Optimization package for Infinite_Money trading system.
"""

from .wasserstein_kelly import WassersteinKellyOptimizer, KellyConstraints

__all__ = [
    "WassersteinKellyOptimizer",
    "KellyConstraints"
]