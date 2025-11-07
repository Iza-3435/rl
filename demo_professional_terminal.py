#!/usr/bin/env python3
"""
Demo of professional terminal output.
Shows what the trading terminal looks like with color-coded trades.
"""

import asyncio
import random
import time
from datetime import datetime
from src.core.terminal_formatter import TerminalFormatter
from src.core.trade_logger import get_trade_logger


async def main():
    """Run trading simulation demo."""
    formatter = TerminalFormatter(use_colors=True)
    trade_logger = get_trade_logger()

    # Show banner
    print(formatter.banner(mode="BALANCED", duration=300, symbols=27))

    # Show initialization
    print(formatter.phase_start("Phase 1", "Market data & network"))
    await asyncio.sleep(0.5)
    print(formatter.phase_complete("Phase 1", 520))

    print(formatter.phase_start("Phase 2", "ML models & routing"))
    await asyncio.sleep(1.0)
    print(formatter.phase_complete("Phase 2", 1050))

    print(formatter.phase_start("Phase 3", "Execution & risk"))
    await asyncio.sleep(0.3)
    print(formatter.phase_complete("Phase 3", 310))

    # System ready
    print(formatter.system_ready())

    # Simulate trades
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "SPY", "QQQ", "JPM", "BAC"]
    sides = ["BUY", "SELL"]
    venues = ["NYSE", "NASDAQ", "IEX", "CBOE", "ARCA"]

    print("")  # Spacing before trades

    # Simulate 20 trades
    for i in range(20):
        symbol = random.choice(symbols)
        side = random.choice(sides)
        quantity = random.randint(50, 200)
        price = random.uniform(100, 500)
        pnl = random.uniform(-50, 80)
        venue = random.choice(venues)

        trade_logger.log_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            venue=venue,
        )

        # Random delay between trades
        await asyncio.sleep(random.uniform(0.1, 0.5))

    # Close table
    trade_logger.print_footer()

    # Show summary
    stats = trade_logger.get_summary_stats()
    print(
        formatter.summary(
            duration=10.5,
            trades=stats["trade_count"],
            total_pnl=stats["total_pnl"],
            win_rate=stats["win_rate"],
            sharpe=1.85,
            max_dd=-2.3,
        )
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PROFESSIONAL TERMINAL OUTPUT DEMO")
    print("  This is how your HFT system will look with the new terminal")
    print("=" * 70 + "\n")
    time.sleep(1)

    asyncio.run(main())
