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

    # Bright colors
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_CYAN = "\033[96m"

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_GRAY = "\033[100m"
    BG_DARK_GRAY = "\033[48;5;235m"
    BG_GREEN = "\033[42m"
    BG_BRIGHT_GREEN = "\033[102m"


class TerminalFormatter:
    """Format trading output for professional terminal display."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.row_alternate = False  # Track alternating rows

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{Color.RESET}"

    def banner(self, mode: str, duration: int, symbols: int) -> str:
        """Generate startup banner."""
        lines = [
            "",
            self._color("╔" + "═" * 68 + "╗", Color.BRIGHT_GREEN),
            self._color("║" + "  HFT NETWORK OPTIMIZER".center(68) + "║", Color.BRIGHT_GREEN + Color.BOLD),
            self._color("║" + "  Production Trading System".center(68) + "║", Color.GREEN),
            self._color("╠" + "═" * 68 + "╣", Color.BRIGHT_GREEN),
            self._color("║", Color.BRIGHT_GREEN) + f"  Mode: {self._color(mode.upper(), Color.YELLOW)}  │  Duration: {self._color(f'{duration}s', Color.WHITE)}  │  Symbols: {self._color(str(symbols), Color.WHITE)}".ljust(68) + self._color("║", Color.BRIGHT_GREEN),
            self._color("╚" + "═" * 68 + "╝", Color.BRIGHT_GREEN),
            ""
        ]
        return "\n".join(lines)

    def phase_start(self, phase: str, description: str) -> str:
        """Format phase initialization message."""
        return f"{self._color('▸', Color.BRIGHT_GREEN)} {phase}: {self._color(description, Color.DIM)}"

    def phase_complete(self, phase: str, duration_ms: Optional[float] = None) -> str:
        """Format phase completion message."""
        time_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        return f"{self._color('✓', Color.GREEN)} {phase} ready{self._color(time_str, Color.GRAY)}"

    def system_ready(self) -> str:
        """Format system ready message."""
        return (
            f"\n{self._color('╔' + '═' * 68 + '╗', Color.BRIGHT_GREEN)}\n"
            f"{self._color('║', Color.BRIGHT_GREEN)}{self._color('  SYSTEM READY', Color.BOLD + Color.BRIGHT_GREEN)} {self._color('│ Trading Active', Color.GREEN)}{''.ljust(42)}{self._color('║', Color.BRIGHT_GREEN)}\n"
            f"{self._color('╚' + '═' * 68 + '╝', Color.BRIGHT_GREEN)}\n"
        )

    def trade_header(self) -> str:
        """Format trade table header."""
        header_line = (
            f"{self._color('Time', Color.WHITE + Color.BOLD):>12}  "
            f"{self._color('Symbol', Color.WHITE + Color.BOLD):<8}  "
            f"{self._color('Side', Color.WHITE + Color.BOLD):<4}  "
            f"{self._color('Qty', Color.WHITE + Color.BOLD):>5}  "
            f"{self._color('Price', Color.WHITE + Color.BOLD):>10}  "
            f"{self._color('P&L', Color.WHITE + Color.BOLD):>12}  "
            f"{self._color('Total P&L', Color.WHITE + Color.BOLD):>13}  "
            f"{self._color('Venue', Color.WHITE + Color.BOLD):<8}"
        )
        top_border = self._color("╔" + "═" * 94 + "╗", Color.BRIGHT_GREEN)
        bottom_border = self._color("╟" + "─" * 94 + "╢", Color.GREEN)
        return f"{top_border}\n{self._color('║', Color.GREEN)} {header_line} {self._color('║', Color.GREEN)}\n{bottom_border}"

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

        # Alternate row background
        self.row_alternate = not self.row_alternate
        bg = Color.BG_DARK_GRAY if self.row_alternate else ""

        # Format timestamp
        if timestamp is None:
            timestamp = datetime.now()
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]

        # Color-code side with background
        if side.upper() in ("BUY", "LONG"):
            side_str = f"{bg}{self._color(f'{side.upper():<4}', Color.CYAN)}"
        else:
            side_str = f"{bg}{self._color(f'{side.upper():<4}', Color.MAGENTA)}"

        # Color-code P&L
        if pnl > 0:
            pnl_str = f"{bg}{self._color(f'+${pnl:>10,.2f}', Color.BRIGHT_GREEN + Color.BOLD)}"
        elif pnl < 0:
            pnl_str = f"{bg}{self._color(f'-${abs(pnl):>10,.2f}', Color.RED)}"
        else:
            pnl_str = f"{bg}{self._color(f' ${pnl:>10,.2f}', Color.GRAY)}"

        # Color-code total P&L
        if total_pnl > 0:
            total_str = f"{bg}{self._color(f'${total_pnl:>12,.2f}', Color.BRIGHT_GREEN + Color.BOLD)}"
        elif total_pnl < 0:
            total_str = f"{bg}{self._color(f'${total_pnl:>12,.2f}', Color.RED + Color.BOLD)}"
        else:
            total_str = f"{bg}{self._color(f'${total_pnl:>12,.2f}', Color.WHITE)}"

        venue_str = (venue or "N/A")[:8]

        # Build row with border and background
        border = self._color("║", Color.GREEN)
        row = (
            f"{border}{bg} "
            f"{bg}{self._color(time_str, Color.GRAY):>12}  "
            f"{bg}{self._color(symbol, Color.WHITE):<8}  "
            f"{side_str}  "
            f"{bg}{self._color(f'{quantity:>5}', Color.WHITE)}  "
            f"{bg}{self._color(f'${price:>9,.2f}', Color.WHITE)}  "
            f"{pnl_str}  "
            f"{total_str}  "
            f"{bg}{self._color(venue_str, Color.GRAY):<8}"
            f"{Color.RESET} {border}"
        )

        return row

    def trade_footer(self) -> str:
        """Format trade table footer."""
        return self._color("╚" + "═" * 94 + "╝", Color.BRIGHT_GREEN)

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
            wr_color = Color.BRIGHT_GREEN
        elif win_rate >= 50:
            wr_color = Color.YELLOW
        else:
            wr_color = Color.RED

        # Color-code total P&L
        if total_pnl > 0:
            pnl_color = Color.BRIGHT_GREEN + Color.BOLD
            pnl_prefix = "+"
        elif total_pnl < 0:
            pnl_color = Color.RED + Color.BOLD
            pnl_prefix = ""
        else:
            pnl_color = Color.WHITE
            pnl_prefix = " "

        lines = [
            "",
            self._color("╔" + "═" * 68 + "╗", Color.BRIGHT_GREEN),
            self._color("║" + "  TRADING SUMMARY".center(68) + "║", Color.BRIGHT_GREEN + Color.BOLD),
            self._color("╠" + "═" * 68 + "╣", Color.BRIGHT_GREEN),
            self._color("║", Color.GREEN) + f"  Duration:        {self._color(f'{duration:.0f}s', Color.WHITE)}".ljust(68) + self._color("║", Color.GREEN),
            self._color("║", Color.GREEN) + f"  Total Trades:    {self._color(str(trades), Color.WHITE)} "
            f"({self._color(f'{self.win_count}W', Color.BRIGHT_GREEN)} / "
            f"{self._color(f'{self.loss_count}L', Color.RED)})".ljust(68) + self._color("║", Color.GREEN),
            self._color("║", Color.GREEN) + f"  Win Rate:        {self._color(f'{win_rate:.1f}%', wr_color)}".ljust(68) + self._color("║", Color.GREEN),
            self._color("║", Color.GREEN) + f"  Total P&L:       {self._color(f'{pnl_prefix}${abs(total_pnl):,.2f}', pnl_color)}".ljust(68) + self._color("║", Color.GREEN),
        ]

        if sharpe is not None:
            sharpe_color = Color.BRIGHT_GREEN if sharpe > 1.0 else Color.YELLOW if sharpe > 0 else Color.RED
            lines.append(self._color("║", Color.GREEN) + f"  Sharpe Ratio:    {self._color(f'{sharpe:.2f}', sharpe_color)}".ljust(68) + self._color("║", Color.GREEN))

        if max_dd is not None:
            lines.append(self._color("║", Color.GREEN) + f"  Max Drawdown:    {self._color(f'{max_dd:.2f}%', Color.RED)}".ljust(68) + self._color("║", Color.GREEN))

        lines.extend([
            self._color("╚" + "═" * 68 + "╝", Color.BRIGHT_GREEN),
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
