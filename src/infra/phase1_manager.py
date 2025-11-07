"""Phase 1: Market data and network infrastructure management."""

from typing import Dict, List
from src.core.types import VenueConfig
from src.core.logging_config import get_logger

logger = get_logger()


class Phase1Manager:
    """Manages market data and network infrastructure."""

    def __init__(
        self, symbols: List[str], venues: Dict[str, VenueConfig], mode: str = "production"
    ):
        self.symbols = symbols
        self.venues = venues
        self.mode = mode

        self.market_generator = None
        self.network_simulator = None
        self.order_book_manager = None
        self.feature_extractor = None
        self.performance_tracker = None

    async def initialize(self):
        """Initialize Phase 1 components."""
        logger.info("Initializing market data and network infrastructure")

        try:
            await self._init_market_data()
            await self._init_network()
            await self._init_order_books()
            await self._init_features()
            await self._init_tracking()

            logger.info("Phase 1 initialization complete")
        except Exception as e:
            logger.error(f"Phase 1 initialization failed: {e}")
            raise

    async def _init_market_data(self):
        """Initialize market data generator."""
        from data.real_market_data_generator import UltraRealisticMarketDataGenerator

        self.market_generator = UltraRealisticMarketDataGenerator(self.symbols, mode=self.mode)
        self.market_generator.venues = self.venues

        logger.verbose(
            "Market data generator initialized",
            symbols=len(self.symbols),
            mode=self.mode,
            tick_rate=self.market_generator.target_ticks_per_minute,
        )

    async def _init_network(self):
        """Initialize network latency simulator."""
        from integration.real_network_system import RealNetworkLatencySimulator

        self.network_simulator = RealNetworkLatencySimulator()
        logger.verbose("Network simulator initialized")

    async def _init_order_books(self):
        """Initialize order book manager."""
        from simulator.order_book_manager import OrderBookManager

        self.order_book_manager = OrderBookManager(self.symbols, self.venues)
        logger.verbose("Order book manager initialized")

    async def _init_features(self):
        """Initialize feature extractor."""
        from data.feature_extractor import FeatureExtractor

        self.feature_extractor = FeatureExtractor(self.symbols, self.venues)
        logger.verbose("Feature extractor initialized")

    async def _init_tracking(self):
        """Initialize performance tracker."""
        from simulator.performance_tracker import PerformanceTracker

        self.performance_tracker = PerformanceTracker()
        logger.verbose("Performance tracker initialized")

    def get_components(self) -> dict:
        """Get all Phase 1 components."""
        return {
            "market_generator": self.market_generator,
            "network_simulator": self.network_simulator,
            "order_book_manager": self.order_book_manager,
            "feature_extractor": self.feature_extractor,
            "performance_tracker": self.performance_tracker,
        }
