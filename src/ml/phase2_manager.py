"""Phase 2: ML latency prediction and routing optimization."""

from typing import Dict, List
from src.core.types import VenueConfig, TradingMode
from src.core.logging_config import get_logger

logger = get_logger()


class Phase2Manager:
    """Manages ML models for latency prediction and routing."""

    def __init__(self, venues: Dict[str, VenueConfig], mode: TradingMode):
        self.venues = venues
        self.mode = mode

        self.latency_predictor = None
        self.ensemble_model = None
        self.routing_environment = None
        self.market_regime_detector = None
        self.online_learner = None

    async def initialize(self, phase1_components: dict):
        """Initialize Phase 2 ML components."""
        logger.info("Initializing ML prediction and routing")

        try:
            await self._init_prediction_models()
            await self._init_routing(phase1_components)
            await self._init_learning_systems()

            logger.info("Phase 2 initialization complete")
        except Exception as e:
            logger.error(f"Phase 2 initialization failed: {e}")
            raise

    async def _init_prediction_models(self):
        """Initialize latency prediction models."""
        try:
            from data.latency_predictor import LatencyPredictor
            from models.ensemble_latency_model import EnsembleLatencyModel

            venue_list = list(self.venues.keys())
            self.latency_predictor = LatencyPredictor(venue_list)
            self.ensemble_model = EnsembleLatencyModel(venue_list)

            self._configure_model_params()
            logger.verbose("Latency prediction models initialized")

        except ImportError as e:
            logger.warning(f"ML models not available: {e}")
            raise

    def _configure_model_params(self):
        """Configure model parameters based on trading mode."""
        params = {
            TradingMode.FAST: {"sequence_length": 10, "update_threshold": 10},
            TradingMode.BALANCED: {"sequence_length": 30, "update_threshold": 25},
            TradingMode.PRODUCTION: {"sequence_length": 50, "update_threshold": 100},
        }

        config = params.get(self.mode, params[TradingMode.PRODUCTION])

        if hasattr(self.latency_predictor, "sequence_length"):
            self.latency_predictor.sequence_length = config["sequence_length"]
            self.latency_predictor.update_threshold = config["update_threshold"]

        logger.verbose(f"Model parameters configured for {self.mode.value}", **config)

    async def _init_routing(self, phase1_components: dict):
        """Initialize RL routing environment."""
        from models.rl_route_optimizer import RoutingEnvironment

        self.routing_environment = RoutingEnvironment(
            self.latency_predictor,
            phase1_components["market_generator"],
            phase1_components["network_simulator"],
            phase1_components["order_book_manager"],
            phase1_components["feature_extractor"],
            venue_list=list(self.venues.keys()),
        )
        logger.verbose("Routing environment initialized")

    async def _init_learning_systems(self):
        """Initialize regime detection and online learning."""
        try:
            from data.logs.market_regime_detector import MarketRegimeDetector, OnlineLearner

            self.market_regime_detector = MarketRegimeDetector()
            self.online_learner = OnlineLearner(
                {
                    "latency_predictor": self.latency_predictor,
                    "ensemble_model": self.ensemble_model,
                    "routing_environment": self.routing_environment,
                }
            )
            logger.verbose("Learning systems initialized")

        except ImportError:
            logger.warning("Market regime detection unavailable")

    def get_components(self) -> dict:
        """Get all Phase 2 components."""
        return {
            "latency_predictor": self.latency_predictor,
            "ensemble_model": self.ensemble_model,
            "routing_environment": self.routing_environment,
            "market_regime_detector": self.market_regime_detector,
            "online_learner": self.online_learner,
        }
