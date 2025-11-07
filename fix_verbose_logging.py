#!/usr/bin/env python3
"""Fix verbose logging across the codebase - Jane Street/Citadel quality."""

import re
import sys
from pathlib import Path

# Emoji removal pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U00002600-\U000027BF"  # Misc symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)


def should_be_debug_level(line: str) -> bool:
    """Determine if a log line should be DEBUG level."""
    debug_keywords = [
        "Updating",
        "REAL ",
        "spread:",
        "liquidity:",
        "arbitrage opportunities",
        "breakdown:",
        "Total slippage:",
        "extracted:",
        "routing:",
    ]
    return any(kw in line for kw in debug_keywords)


def fix_file(filepath: Path) -> tuple[int, int, int]:
    """
    Fix logging in a single file.
    Returns: (print_statements_removed, emojis_removed, lines_modified)
    """
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0, 0, 0

    lines = content.split("\n")
    modified_lines = []
    prints_removed = 0
    emojis_removed = 0
    lines_changed = 0
    needs_logger_import = False

    for line in lines:
        original_line = line

        # Remove emojis
        if EMOJI_PATTERN.search(line):
            line = EMOJI_PATTERN.sub("", line)
            emojis_removed += 1

        # Convert print statements to logger calls
        if "print(" in line and not line.strip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()

            if stripped.startswith("print("):
                # Extract print content
                match = re.match(r'print\((.*)\)', stripped)
                if match:
                    content_str = match.group(1)

                    # Determine log level
                    if should_be_debug_level(content_str):
                        new_line = " " * indent + f"logger.debug({content_str})"
                    else:
                        new_line = " " * indent + f"logger.info({content_str})"

                    line = new_line
                    prints_removed += 1
                    needs_logger_import = True

        # Clean up redundant phrases
        line = line.replace("ENHANCED ", "")
        line = line.replace("REGULAR ", "")
        line = line.replace("GENERATED ", "Generated ")
        line = line.replace("EXECUTING ", "Executing ")
        line = line.replace("EXECUTED:", ":")

        if line != original_line:
            lines_changed += 1

        modified_lines.append(line)

    # Add logger import if needed
    if needs_logger_import and "import logging" not in content:
        # Find where to insert logging import
        for i, line in enumerate(modified_lines):
            if line.startswith("import ") or line.startswith("from "):
                # Insert after other imports
                if i + 1 < len(modified_lines) and not (
                    modified_lines[i + 1].startswith("import")
                    or modified_lines[i + 1].startswith("from")
                ):
                    modified_lines.insert(i + 1, "import logging")
                    modified_lines.insert(i + 2, "")
                    modified_lines.insert(i + 3, "logger = logging.getLogger(__name__)")
                    modified_lines.insert(i + 4, "")
                    break

    # Write back
    filepath.write_text("\n".join(modified_lines))

    return prints_removed, emojis_removed, lines_changed


def main():
    """Fix all verbose logging in the codebase."""
    root = Path("/home/user/rl")

    # Files to fix (in priority order)
    files_to_fix = [
        "data/real_market_data_generator.py",
        "data/advanced_technical_indicators.py",
        "src/integration/phase3/component_initializers.py",
        "src/integration/phase3/training_manager.py",
        "src/integration/phase3/execution_pipeline.py",
        "monitoring/system_health_monitor.py",
    ]

    total_prints = 0
    total_emojis = 0
    total_lines = 0

    print("=" * 70)
    print("FIXING VERBOSE LOGGING - JANE STREET/CITADEL QUALITY")
    print("=" * 70)

    for filepath_str in files_to_fix:
        filepath = root / filepath_str
        if not filepath.exists():
            print(f"  SKIP: {filepath_str} (not found)")
            continue

        prints, emojis, lines = fix_file(filepath)
        total_prints += prints
        total_emojis += emojis
        total_lines += lines

        print(f"  ✓ {filepath_str}")
        print(f"      {prints} print() → logger, {emojis} emojis removed, {lines} lines modified")

    print("=" * 70)
    print(f"TOTAL: {total_prints} prints fixed, {total_emojis} emojis removed, {total_lines} lines modified")
    print("=" * 70)
    print("\nNow use: python main.py --log-level quiet")


if __name__ == "__main__":
    main()
