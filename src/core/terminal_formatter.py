"""Professional trading terminal output formatter."""

from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class Color:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background
    BG_RED = "\033[101m"
    BG_GREEN = "\033[102m"


class TerminalFormatter:
    """Format trading output for professional terminal display."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{Color.RESET}"

    def banner(self, mode: str, duration: int, symbols: int) -> str:
        """Generate startup banner."""
        lines = [
            "",
            self._color("═" * 70, Color.CYAN),
            self._color("  HFT NETWORK OPTIMIZER", Color.BOLD) + self._color(" | Production Trading System", Color.CYAN),
            self._color("═" * 70, Color.CYAN),
            f"  Mode: {self._color(mode.upper(), Color.YELLOW)}  |  Duration: {self._color(f'{duration}s', Color.WHITE)}  |  Symbols: {self._color(str(symbols), Color.WHITE)}",
            self._color("─" * 70, Color.GRAY),
            ""
        ]
        return "\n".join(lines)

    def phase_start(self, phase: str, description: str) -> str:
        """Format phase initialization message."""
        return f"{self._color('▸', Color.CYAN)} {phase}: {self._color(description, Color.DIM)}"

    def phase_complete(self, phase: str, duration_ms: Optional[float] = None) -> str:
        """Format phase completion message."""
        time_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        return f"{self._color('✓', Color.GREEN)} {phase} ready{self._color(time_str, Color.GRAY)}"

    def system_ready(self) -> str:
        """Format system ready message."""
        return (
            f"\n{self._color('━' * 70, Color.GREEN)}\n"
            f"{self._color('  SYSTEM READY', Color.BOLD + Color.GREEN)} {self._color('| Trading active', Color.GREEN)}\n"
            f"{self._color('━' * 70, Color.GREEN)}\n"
        )

    def trade_header(self) -> str:
        """Format trade table header."""
        header = (
            f"{self._color('Time', Color.GRAY):>12}  "
            f"{self._color('Symbol', Color.GRAY):<6}  "
            f"{self._color('Side', Color.GRAY):<4}  "
            f"{self._color('Qty', Color.GRAY):>5}  "
            f"{self._color('Price', Color.GRAY):>10}  "
            f"{self._color('P&L', Color.GRAY):>12}  "
            f"{self._color('Total P&L', Color.GRAY):>13}  "
            f"{self._color('Venue', Color.GRAY):<8}"
        )
        sep = self._color("─" * 70, Color.GRAY)
        return f"{header}\n{sep}"

    def trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        pnl: float,
        total_pnl: float,
        venue: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Format a single trade execution."""
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1

        # Format timestamp
        if timestamp is None:
            timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]

        # Color-code side
        if side.upper() in ("BUY", "LONG"):
            side_str = self._color(f"{side.upper():<4}", Color.BLUE)
        else:
            side_str = self._color(f"{side.upper():<4}", Color.MAGENTA)

        # Color-code P&L
        if pnl > 0:
            pnl_str = self._color(f"+${pnl:>10,.2f}", Color.GREEN)
        elif pnl < 0:
            pnl_str = self._color(f"-${abs(pnl):>10,.2f}", Color.RED)
        else:
            pnl_str = self._color(f" ${pnl:>10,.2f}", Color.GRAY)

        # Color-code total P&L
        if total_pnl > 0:
            total_str = self._color(f"${total_pnl:>12,.2f}", Color.GREEN + Color.BOLD)
        elif total_pnl < 0:
            total_str = self._color(f"${total_pnl:>12,.2f}", Color.RED + Color.BOLD)
        else:
            total_str = self._color(f"${total_pnl:>12,.2f}", Color.WHITE)

        venue_str = (venue or "N/A")[:8]

        return (
            f"{self._color(time_str, Color.GRAY):>12}  "
            f"{symbol:<6}  "
            f"{side_str}  "
            f"{quantity:>5}  "
            f"{self._color(f'${price:>9,.2f}', Color.WHITE)}  "
            f"{pnl_str}  "
            f"{total_str}  "
            f"{self._color(venue_str, Color.GRAY):<8}"
        )

    def training_progress(self, current: int, target: int, elapsed: float) -> str:
        """Format training progress."""
        pct = (current / target * 100) if target > 0 else 0
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        return (
            f"\r{self._color('Training:', Color.CYAN)} "
            f"{self._color(bar, Color.GREEN)} "
            f"{self._color(f'{pct:>5.1f}%', Color.WHITE)} "
            f"{self._color(f'({current:,}/{target:,} ticks)', Color.GRAY)} "
            f"{self._color(f'{elapsed:.0f}s', Color.GRAY)}"
        )

    def summary(
        self,
        duration: float,
        trades: int,
        total_pnl: float,
        win_rate: float,
        sharpe: Optional[float] = None,
        max_dd: Optional[float] = None,
    ) -> str:
        """Format final trading summary."""

        # Color-code win rate
        if win_rate >= 60:
            wr_color = Color.GREEN
        elif win_rate >= 50:
            wr_color = Color.YELLOW
        else:
            wr_color = Color.RED

        # Color-code total P&L
        if total_pnl > 0:
            pnl_color = Color.GREEN + Color.BOLD
            pnl_prefix = "+"
        elif total_pnl < 0:
            pnl_color = Color.RED + Color.BOLD
            pnl_prefix = ""
        else:
            pnl_color = Color.WHITE
            pnl_prefix = " "

        lines = [
            "",
            self._color("═" * 70, Color.CYAN),
            self._color("  TRADING SUMMARY", Color.BOLD + Color.CYAN),
            self._color("═" * 70, Color.CYAN),
            f"  Duration:        {self._color(f'{duration:.0f}s', Color.WHITE)}",
            f"  Total Trades:    {self._color(str(trades), Color.WHITE)} "
            f"({self._color(f'{self.win_count}W', Color.GREEN)} / "
            f"{self._color(f'{self.loss_count}L', Color.RED)})",
            f"  Win Rate:        {self._color(f'{win_rate:.1f}%', wr_color)}",
            f"  Total P&L:       {self._color(f'{pnl_prefix}${abs(total_pnl):,.2f}', pnl_color)}",
        ]

        if sharpe is not None:
            sharpe_color = Color.GREEN if sharpe > 1.0 else Color.YELLOW if sharpe > 0 else Color.RED
            lines.append(f"  Sharpe Ratio:    {self._color(f'{sharpe:.2f}', sharpe_color)}")

        if max_dd is not None:
            lines.append(f"  Max Drawdown:    {self._color(f'{max_dd:.2f}%', Color.RED)}")

        lines.extend([
            self._color("═" * 70, Color.CYAN),
            ""
        ])

        return "\n".join(lines)

    def error(self, message: str) -> str:
        """Format error message."""
        return f"{self._color('✗', Color.RED)} {self._color(message, Color.RED)}"

    def warning(self, message: str) -> str:
        """Format warning message."""
        return f"{self._color('⚠', Color.YELLOW)} {self._color(message, Color.YELLOW)}"

    def info(self, message: str) -> str:
        """Format info message."""
        return f"{self._color('ℹ', Color.BLUE)} {message}"

    def metric(self, label: str, value: str, good: bool = True) -> str:
        """Format a metric display."""
        color = Color.GREEN if good else Color.RED
        return f"  {label:<20} {self._color(value, color)}"
