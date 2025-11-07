# HFT Network Optimizer

Production-grade high-frequency trading system with ML-based latency prediction and optimal routing.

## Architecture

**3-Phase Modular Design:**

- **Phase 1**: Market data infrastructure and network latency measurement
- **Phase 2**: ML-based latency prediction and reinforcement learning routing optimization
- **Phase 3**: Trade execution, risk management, and backtesting

## Features

- Real-time market data processing from multiple venues
- Deep learning latency prediction (DQN, PPO)
- Multi-venue routing optimization
- Comprehensive risk management
- Sub-millisecond execution targeting
- Production-grade logging and monitoring
- Backtesting framework

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd rl

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create `config/production.yaml`:

```yaml
trading:
  symbols:
    - AAPL
    - MSFT
    - GOOGL
  max_position_size: 10000
  max_portfolio_value: 1000000

system:
  mode: production
  log_level: normal
```

### Run

```bash
# Production mode (recommended)
python main.py --mode production --duration 600

# Fast mode (2-minute demo)
python main.py --mode fast --duration 120 --log-level verbose

# Custom symbols
python main.py --symbols AAPL,TSLA,NVDA --duration 300
```

### CLI Options

```
--mode              Trading mode: fast, balanced, production (default: production)
--duration          Duration in seconds (default: 600)
--symbols           Comma-separated symbols to trade
--config            Path to config file
--log-level         Logging: quiet, normal, verbose, debug (default: normal)
--skip-backtest     Skip backtesting phase
```

## Project Structure

```
rl/
├── src/
│   ├── core/              # Core types, config, logging
│   ├── infra/             # Phase 1: Market data, network
│   ├── ml/                # Phase 2: ML models, routing
│   ├── execution/         # Phase 3: Trading, risk
│   └── strategies/        # Trading strategies
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance benchmarks
├── config/                # Configuration files
├── data/                  # Legacy data modules
├── models/                # Legacy ML modules
├── simulator/             # Legacy simulator modules
├── main.py                # Entry point
└── requirements.txt       # Dependencies
```

## Development

### Testing

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_config.py -v
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy src/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Docker

### Build

```bash
docker-compose build
```

### Run

```bash
# Production
docker-compose up hft-system

# With monitoring stack
docker-compose up
```

### Services

- **hft-system**: Main trading system (port 8080)
- **prometheus**: Metrics (port 9090)
- **grafana**: Dashboards (port 3000)
- **redis**: Caching (port 6379)

## Monitoring

### Prometheus Metrics

- Trade execution latency (p50, p95, p99)
- Order fill rates
- P&L tracking
- Risk limit utilization

### Grafana Dashboards

Access at `http://localhost:3000` (admin/admin)

- Real-time P&L
- Latency heatmaps
- Venue performance comparison
- Risk metrics

## Configuration

### Environment Variables

```bash
TRADING_MODE=production
TRADING_SYMBOLS=AAPL,MSFT,GOOGL
MAX_POSITION_SIZE=10000
LOG_LEVEL=normal
FINNHUB_API_KEY=your_key_here
```

### Logging Levels

- **quiet**: Errors only
- **normal**: Key milestones (default)
- **verbose**: Detailed progress
- **debug**: All debug information

## Performance

**Target Metrics:**

- Order-to-execution latency: < 1ms (p95)
- Tick processing: > 10,000/sec
- ML prediction latency: < 100μs
- System uptime: 99.9%

## Risk Management

**Built-in Controls:**

- Position size limits
- Portfolio value caps
- Real-time P&L monitoring
- Cross-venue price validation
- Automatic circuit breakers

## API Keys

Required for market data:

- **Finnhub**: Real-time quotes (free tier available)
- **Yahoo Finance**: Historical data (via yfinance, no key needed)

Store in `.env` file, never commit actual keys.

## Troubleshooting

### Import Errors

```bash
# Ensure src/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Dependencies

```bash
# Reinstall clean
pip install --upgrade --force-reinstall -r requirements.txt
```

### Performance Issues

- Use `--mode fast` for testing
- Reduce number of symbols
- Check system resources with `docker stats`

## Contributing

1. Follow PEP 8 style guide
2. Add tests for new features
3. Keep files under 500 lines
4. Use type hints
5. Write professional docstrings (no AI-style)

## License

Proprietary - Internal Use Only

## Support

For issues or questions, contact the trading systems team.
