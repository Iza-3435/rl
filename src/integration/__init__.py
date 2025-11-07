"""System integration components."""

from .phase3 import (
    ProductionExecutionPipeline,
    IntegratedMLPredictor,
    ComponentInitializer,
    TrainingManager,
    EXPANDED_STOCK_LIST,
    get_venue_configs,
)

try:
    from integration.phase3_complete_integration import Phase3CompleteIntegration
except ImportError:
    pass

__all__ = [
    "Phase3CompleteIntegration",
    "ProductionExecutionPipeline",
    "IntegratedMLPredictor",
    "ComponentInitializer",
    "TrainingManager",
    "EXPANDED_STOCK_LIST",
    "get_venue_configs",
]
