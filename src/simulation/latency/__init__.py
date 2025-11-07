"""Latency simulation system.

NOTE: Full refactor in progress. Currently imports from legacy file.
Target: Break 1,364 LOC into 6-8 modules of <200 LOC each.

Completed modules:
- types.py: Core types and dataclasses (89 LOC)
- message_queue.py: Queue simulation (67 LOC)

Pending refactor:
- LatencySimulator (605 LOC) -> Split into 3 modules
- EnhancedOrderExecutionEngine (438 LOC) -> Split into 2 modules
- LatencyAnalytics (140 LOC) -> Keep as module
"""

from .types import (
    LatencyComponent,
    CongestionLevel,
    LatencyBreakdown,
    VenueLatencyProfile,
)
from .message_queue import MessageQueue

try:
    from simulator.enhanced_latency_simulation import (
        LatencySimulator,
        EnhancedOrderExecutionEngine,
        LatencyAnalytics,
    )
except ImportError:
    pass

__all__ = [
    "LatencyComponent",
    "CongestionLevel",
    "LatencyBreakdown",
    "VenueLatencyProfile",
    "MessageQueue",
    "LatencySimulator",
    "EnhancedOrderExecutionEngine",
    "LatencyAnalytics",
]
