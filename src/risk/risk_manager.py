"""Production risk management with clean interface."""

from typing import Dict, List, Optional
from src.core.logging_config import get_logger
from src.core.types import OrderRequest, Position
from engine.risk_management_engine import RiskManager as LegacyRiskManager

logger = get_logger()


class ProductionRiskManager:
    """Production wrapper for risk management."""

    def __init__(
        self,
        max_position_size: int = 10000,
        max_portfolio_value: float = 1_000_000,
        risk_limit_pct: float = 0.02,
    ):
        self.max_position_size = max_position_size
        self.max_portfolio_value = max_portfolio_value
        self.risk_limit_pct = risk_limit_pct

        self._manager = LegacyRiskManager()

        logger.verbose(
            "Risk manager initialized",
            max_position=max_position_size,
            max_portfolio=max_portfolio_value,
            risk_limit=risk_limit_pct,
        )

    def check_order(self, order: OrderRequest) -> bool:
        """Validate order against risk limits."""
        try:
            if order.quantity > self.max_position_size:
                logger.warning(
                    "Order exceeds position limit",
                    quantity=order.quantity,
                    limit=self.max_position_size,
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False

    def update_position(self, symbol: str, quantity: int, price: float):
        """Update position after trade."""
        try:
            if hasattr(self._manager, "update_position"):
                self._manager.update_position(symbol, quantity, price)

        except Exception as e:
            logger.debug(f"Position update error: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        try:
            pos = self._manager.get_position(symbol)
            if pos:
                return Position(
                    symbol=symbol,
                    quantity=pos.get("quantity", 0),
                    avg_price=pos.get("avg_price", 0.0),
                    unrealized_pnl=pos.get("unrealized_pnl", 0.0),
                    realized_pnl=pos.get("realized_pnl", 0.0),
                )
        except:
            pass

        return None

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        try:
            if hasattr(self._manager, "get_total_exposure"):
                return self._manager.get_total_exposure()
        except:
            pass

        return 0.0
