"""Execution simulation system."""

from .fees import DEFAULT_FEE_SCHEDULE

try:
    from simulator.enhanced_latency_simulation import (
        EnhancedOrderExecutionEngine,
        LatencyAnalytics,
    )
except ImportError:
    pass

__all__ = [
    "DEFAULT_FEE_SCHEDULE",
    "EnhancedOrderExecutionEngine",
    "LatencyAnalytics",
]
