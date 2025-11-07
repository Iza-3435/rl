#!/usr/bin/env python3
"""Fix log levels to respect quiet/normal/verbose/debug modes."""

import re
from pathlib import Path


def fix_log_levels(filepath: Path) -> int:
    """Fix log levels in file to be appropriate."""
    content = filepath.read_text()
    lines = content.split("\n")
    modified = []
    changes = 0

    # Keywords that should be DEBUG level (verbose details)
    debug_keywords = [
        "Updating",
        "symbols:",
        "REAL ",
        "spread:",
        "liquidity:",
        "arbitrage opportunities",
        "breakdown:",
        "Total slippage:",
        "extracted:",
        "routing:",
        "tick counts:",
        "Actual rate:",
        "Rate efficiency:",
        "Per-symbol",
        "multipliers:",
        "update interval:",
        "Target Rate:",
        "Real venues:",
    ]

    # Keywords that should use logger.verbose() (show in verbose/debug)
    verbose_keywords = [
        "initialized",
        "Generator initialized",
        "Starting",
        "complete!",
        "Total ticks:",
        "Target tick rate:",
    ]

    for line in lines:
        original = line

        # Remove any remaining print() statements
        if "print(" in line and not line.strip().startswith("#"):
            # Skip or convert to debug
            if "REAL " in line or "ARBITRAGE" in line or "Updating" in line:
                line = line.replace("print(", "# print(", 1)  # Comment out
                changes += 1

        # Convert logger.info() to appropriate level
        if "logger.info(" in line:
            # Check if it should be debug
            if any(kw in line for kw in debug_keywords):
                line = line.replace("logger.info(", "logger.debug(")
                changes += 1
            # Check if it should be verbose (only in verbose/debug mode)
            elif any(kw in line for kw in verbose_keywords):
                # Use logger.debug for now since ProductionLogger.verbose() exists
                if "ProductionLogger" not in str(filepath):
                    line = line.replace("logger.info(", "logger.debug(")
                    changes += 1

        modified.append(line)

    if changes > 0:
        filepath.write_text("\n".join(modified))

    return changes


def main():
    """Fix log levels across the codebase."""
    root = Path("/home/user/rl")

    files = [
        "data/real_market_data_generator.py",
        "data/advanced_technical_indicators.py",
        "monitoring/system_health_monitor.py",
        "src/integration/phase3/component_initializers.py",
        "src/integration/phase3/training_manager.py",
    ]

    print("=" * 70)
    print("FIXING LOG LEVELS FOR JANE STREET/CITADEL QUALITY")
    print("=" * 70)

    total_changes = 0
    for filepath_str in files:
        filepath = root / filepath_str
        if not filepath.exists():
            print(f"  SKIP: {filepath_str}")
            continue

        changes = fix_log_levels(filepath)
        total_changes += changes
        print(f"  ✓ {filepath_str}: {changes} log levels fixed")

    print("=" * 70)
    print(f"TOTAL: {total_changes} log statements adjusted")
    print("=" * 70)


if __name__ == "__main__":
    main()
