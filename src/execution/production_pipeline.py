"""Production execution pipeline for HFT trading."""

import asyncio
from typing import Dict, Optional
from datetime import datetime

from src.core.types import OrderRequest, Trade, MarketTick
from src.core.logging_config import get_logger

logger = get_logger()


class ProductionExecutionPipeline:
    """Orchestrates complete trade execution pipeline."""

    def __init__(self, phase1_components: dict, phase2_components: dict, phase3_components: dict):
        self.phase1 = phase1_components
        self.phase2 = phase2_components
        self.phase3 = phase3_components

        self.is_running = False
        self.metrics = {
            "ticks_processed": 0,
            "predictions_made": 0,
            "trades_executed": 0,
            "total_pnl": 0.0,
        }

    async def start(self, duration_seconds: int):
        """Start production trading pipeline."""
        logger.info(f"Starting production pipeline for {duration_seconds}s")
        self.is_running = True

        try:
            await self._run_pipeline(duration_seconds)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.is_running = False
            await self._cleanup()

    async def _run_pipeline(self, duration: int):
        """Run main trading loop."""
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < duration and self.is_running:
            try:
                tick = await self._get_market_tick()
                if tick:
                    await self._process_tick(tick)

                await asyncio.sleep(0.001)

            except Exception as e:
                logger.error(f"Tick processing error: {e}")

        logger.info("Pipeline completed", **self.metrics)

    async def _get_market_tick(self) -> Optional[MarketTick]:
        """Get next market data tick."""
        if self.phase1["market_generator"]:
            return await self.phase1["market_generator"].get_next_tick()
        return None

    async def _process_tick(self, tick: MarketTick):
        """Process market tick through pipeline."""
        self.metrics["ticks_processed"] += 1

        features = self._extract_features(tick)
        prediction = await self._predict_latency(features)
        route = self._optimize_route(tick, prediction)
        order = self._generate_order(tick, route)

        if order and self._check_risk(order):
            trade = await self._execute_trade(order)
            if trade:
                await self._update_metrics(trade)

    def _extract_features(self, tick: MarketTick) -> dict:
        """Extract trading features from tick."""
        if self.phase1["feature_extractor"]:
            return self.phase1["feature_extractor"].extract(tick)
        return {}

    async def _predict_latency(self, features: dict) -> dict:
        """Predict venue latencies."""
        if self.phase2["latency_predictor"]:
            self.metrics["predictions_made"] += 1
            return self.phase2["latency_predictor"].predict(features)
        return {}

    def _optimize_route(self, tick: MarketTick, prediction: dict) -> Optional[str]:
        """Optimize routing decision."""
        if self.phase2["routing_environment"]:
            return self.phase2["routing_environment"].get_best_venue(tick, prediction)
        return None

    def _generate_order(self, tick: MarketTick, venue: Optional[str]) -> Optional[OrderRequest]:
        """Generate order from signal."""
        return None

    def _check_risk(self, order: OrderRequest) -> bool:
        """Validate order against risk limits."""
        if self.phase3["risk_manager"]:
            return self.phase3["risk_manager"].check_order(order)
        return True

    async def _execute_trade(self, order: OrderRequest) -> Optional[Trade]:
        """Execute trade through simulator."""
        if self.phase3["trading_simulator"]:
            self.metrics["trades_executed"] += 1
            return await self.phase3["trading_simulator"].execute(order)
        return None

    async def _update_metrics(self, trade: Trade):
        """Update performance metrics."""
        if trade:
            self.metrics["total_pnl"] += trade.price * trade.quantity

    async def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping production pipeline")
        self.is_running = False

    async def _cleanup(self):
        """Cleanup resources."""
        logger.verbose("Cleaning up pipeline resources")
        await asyncio.sleep(0.01)

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()
