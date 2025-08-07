import yfinance as yf
import asyncio
import time
from datetime import datetime
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from typing import List, Dict, Optional, Any, Tuple

# ADD THIS AFTER YOUR IMPORTS (around line 30)
EXPANDED_STOCK_LIST = [
    # Original tech stocks
    'AAPL', 'MSFT', 'GOOGL',
    
    # High-volume tech
    'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX',
    
    # Financial sector  
    'JPM', 'BAC', 'WFC', 'GS', 'C',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV',
    
    # Consumer/Industrial
    'PG', 'KO', 'XOM', 'CVX', 'DIS',
    
    # High-volume ETFs
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT'
]

OPTIMAL_TICK_RATES = {
    'development': {
        'target_ticks_per_minute': 200,   # 3.3 ticks/second
        'training_duration_minutes': 10,   # 2,000 total ticks
        'symbols': 27                      # Focus on key symbols
    },
    'balanced': {
        'target_ticks_per_minute': 400,   # 6.7 ticks/second
        'training_duration_minutes': 15,   # 6,000 total ticks
        'symbols': 27
    },
    'production': {
        'target_ticks_per_minute': 600,   # 10 ticks/second  
        'training_duration_minutes': 30,   # 18,000 total ticks
        'symbols': 27                     
    }
}

@dataclass
class RealMarketTick:
    """Enhanced MarketTick with real-world data"""
    symbol: str
    venue: str
    timestamp: float
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    real_spread: float
    market_cap: Optional[float] = None
    day_change: Optional[float] = None
    volatility: Optional[float] = None
    
    # Real market microstructure
    is_market_hours: bool = True
    exchange_status: str = "open"
    liquidity_tier: str = "high"  # high/medium/low
    
    @property
    def spread(self) -> float:
        return self.real_spread
    
    @property 
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2

@dataclass
class VenueConfig:
    """Configuration for each trading venue"""
    name: str
    base_latency_us: int  # Base latency in microseconds
    jitter_range: Tuple[int, int]  # Min/max jitter in microseconds
    packet_loss_rate: float
    congestion_factor: float 

class EnhancedTickGenerator:
    """Enhanced tick generation for better ML model training"""
    
    def __init__(self, base_update_interval=10):
        self.base_update_interval = base_update_interval
        self.tick_multipliers = {
        # High-frequency symbols (more ticks) - ETFs and most liquid
                'SPY': 8,    # S&P 500 ETF - most liquid
                'QQQ': 7,    # NASDAQ ETF
                'IWM': 6,    # Russell 2000 ETF
                
                # High-volatility tech (frequent updates)
                'TSLA': 6,   # Most volatile large cap
                'NVDA': 6,   # AI/crypto momentum
                'META': 5,   # Social media volatility
                
                # Large cap tech (high volume)
                'AAPL': 5,   # Largest market cap
                'MSFT': 5,   # Second largest
                'GOOGL': 4,  # Alphabet
                'AMZN': 4,   # Amazon
                'NFLX': 4,   # Netflix
                
                # Financials (moderate frequency)
                'JPM': 3,    # JP Morgan
                'BAC': 3,    # Bank of America
                'WFC': 3,    # Wells Fargo
                'GS': 3,     # Goldman Sachs
                'C': 3,      # Citigroup
                
                # Healthcare (stable but important)
                'JNJ': 2,    # Johnson & Johnson
                'PFE': 2,    # Pfizer
                'UNH': 3,    # UnitedHealth (more volatile)
                'ABBV': 2,   # AbbVie
                
                # Consumer/Industrial (moderate)
                'PG': 2,     # Procter & Gamble
                'KO': 2,     # Coca-Cola
                'XOM': 3,    # Exxon (energy volatility)
                'CVX': 3,    # Chevron
                'DIS': 2,    # Disney
                
                # Alternative assets (lower frequency)
                'GLD': 2,    # Gold ETF
                'TLT': 1,    # Treasury bonds (lowest volatility)
            }
                
        self.latency_history = {venue: deque(maxlen=100) for venue in self.tick_multipliers.keys()}
        self.congestion_events = []
        
        print("🔧 Enhanced Tick Generator initialized with multipliers:")
        for symbol, multiplier in self.tick_multipliers.items():
            print(f"   {symbol}: {multiplier}x")

    def get_update_interval(self, symbol):
        """Get update interval for specific symbol"""
        multiplier = self.tick_multipliers.get(symbol, 3)  # Default 3x
        return self.base_update_interval / multiplier
    
    def generate_intraday_ticks(self, symbol, base_data, num_ticks=100):
        """Generate realistic intraday tick sequence"""
        ticks = []
        current_price = base_data['price']
        current_volume = base_data['volume']
        volatility = base_data.get('volatility', 0.02)
        
        # Time-based tick generation
        import datetime
        start_time = time.time()
        
        for i in range(num_ticks):
            # Realistic price movement (mean-reverting with momentum)
            price_change_pct = np.random.normal(0, volatility/np.sqrt(252*390)) 
            
            # Add autocorrelation (momentum)
            if i > 0:
                momentum = (ticks[-1]['price'] - current_price) / current_price * 0.1
                price_change_pct += momentum
            
            new_price = current_price * (1 + price_change_pct)
            
            # Volume clustering (high volume after price moves)
            volume_multiplier = 1 + abs(price_change_pct) * 50
            new_volume = int(current_volume * np.random.uniform(0.5, volume_multiplier))
            
            # Realistic spreads based on volatility and volume
            base_spread_bps = 1.0  # 1 basis point base
            vol_impact = abs(price_change_pct) * 1000  # Higher vol = wider spreads
            volume_impact = max(0.5, min(2.0, 1000000 / new_volume))  # Lower volume = wider spreads
            
            spread_bps = base_spread_bps * (1 + vol_impact) * volume_impact
            spread_dollars = new_price * spread_bps / 10000
            
            tick = {
                'timestamp': start_time + i * self.get_update_interval(symbol),
                'symbol': symbol,
                'price': new_price,
                'bid': new_price - spread_dollars/2,
                'ask': new_price + spread_dollars/2,
                'volume': new_volume,
                'spread_bps': spread_bps
            }
            ticks.append(tick)
            current_price = new_price
        
        return ticks
    
    def get_tick_frequency_for_mode(self, mode='balanced'):
        """Get optimal tick configuration for training mode"""
        config = OPTIMAL_TICK_RATES.get(mode, OPTIMAL_TICK_RATES['balanced'])
        
        # Calculate base interval to achieve target rate
        target_rate = config['target_ticks_per_minute'] / 60  # ticks per second
        base_interval = 1.0 / target_rate
        
        return {
            'base_interval': base_interval,
            'target_ticks_per_minute': config['target_ticks_per_minute'],
            'recommended_symbols': config['symbols'],
            'training_duration': config['training_duration_minutes']
        }
    
    def generate_symbol_priorities(self, symbols, mode='balanced'):
        """Generate priority list of symbols based on tick frequency"""
        priorities = []
        for symbol in symbols:
            multiplier = self.tick_multipliers.get(symbol, 3)
            priorities.append((symbol, multiplier))
        
        # Sort by frequency (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N symbols based on mode
        config = OPTIMAL_TICK_RATES[mode]
        max_symbols = config['symbols']
        
        return [symbol for symbol, _ in priorities[:max_symbols]]

class UltraRealisticMarketDataGenerator:
    """The most realistic market data generator possible - ENHANCED VERSION"""

    def __init__(self, symbols: List[str] = None, mode: str = 'balanced'):
        if symbols is None:
            symbols = EXPANDED_STOCK_LIST
        elif len(symbols) < 5:
            print(f"⚠️  Only {len(symbols)} symbols provided. Consider using more!")
        
        # 🔧 ADD #3: INITIALIZE ENHANCED TICK GENERATOR (ADD TO __init__)
        self.enhanced_tick_gen = EnhancedTickGenerator()
        self.mode = mode
        
        # Get optimal configuration for mode
        tick_config = self.enhanced_tick_gen.get_tick_frequency_for_mode(mode)
        self.base_update_interval = tick_config['base_interval']
        self.target_ticks_per_minute = tick_config['target_ticks_per_minute']
        
        # # Optimize symbol selection for the mode
        # if len(symbols) > tick_config['recommended_symbols']:
        #     priority_symbols = self.enhanced_tick_gen.generate_symbol_priorities(symbols, mode)
        #     self.symbols = priority_symbols
        #     print(f"🎯 OPTIMIZED: Using top {len(self.symbols)} symbols for {mode} mode")
        # else:
        #     self.symbols = symbols
        # Force all symbols regardless of mode
        self.symbols = symbols
        print(f"🎯 FORCED: Using ALL {len(self.symbols)} symbols (ignoring {mode} mode limits)")
        
        print(f"🚀 Enhanced Trading {len(self.symbols)} symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''}")
        print(f"📊 Target tick rate: {self.target_ticks_per_minute} ticks/minute ({self.target_ticks_per_minute/60:.1f}/sec)")
        print(f"⏱️  Base update interval: {self.base_update_interval:.2f} seconds")

        # Your existing initialization code continues...
        self.real_venues = {
            'NYSE': {'endpoint': 'www.nyse.com', 'maker_fee': 0.0003, 'rebate': 0.0001},
            'NASDAQ': {'endpoint': 'www.nasdaq.com', 'maker_fee': 0.0002, 'rebate': 0.0001},
            'ARCA': {'endpoint': 'www.nyse.com', 'maker_fee': 0.0003, 'rebate': 0.0001},
            'IEX': {'endpoint': 'iextrading.com', 'maker_fee': 0.0000, 'rebate': 0.0000},
            'CBOE': {'endpoint': 'www.cboe.com', 'maker_fee': 0.0003, 'rebate': 0.0001}
        }
        
        self.arbitrage_opportunities = deque(maxlen=50)
        self.tick_count = 0
        self.current_prices = {}
        self.market_hours_cache = {}
        
        print("🌟 ENHANCED Market Data Generator initialized")
        print(f"📊 Symbols: {self.symbols}")
        print(f"🏛️  Real venues: {list(self.real_venues.keys())}")

    def _get_time_of_day_factors(self, current_time: datetime) -> Dict[str, float]:
        """Apply realistic time-of-day effects to market data"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Market open surge: 9:30-10:30 AM
        if 9 <= hour < 11:
            spread_multiplier = 1.4  # 40% wider spreads
            latency_multiplier = 1.6  # 60% higher latency
            volume_multiplier = 2.2   # 120% higher volume
        
        # Lunch lull: 12:00-1:00 PM  
        elif 12 <= hour < 13:
            spread_multiplier = 1.1   # 10% wider spreads
            latency_multiplier = 0.8  # 20% lower latency
            volume_multiplier = 0.6   # 40% lower volume
        
        # Market close: 3:30-4:00 PM
        elif hour >= 15 and (hour > 15 or minute >= 30):
            spread_multiplier = 1.5   # 50% wider spreads
            latency_multiplier = 2.1  # 110% higher latency
            volume_multiplier = 3.0   # 200% higher volume
            
        else:  # Regular trading hours
            spread_multiplier = 1.0
            latency_multiplier = 1.0  
            volume_multiplier = 1.0
        
        return {
            'spread_factor': spread_multiplier,
            'latency_factor': latency_multiplier, 
            'volume_factor': volume_multiplier
        }
    
    def _apply_volume_spread_dynamics(self, base_spread: float, volume: int, avg_volume: int) -> float:
        """Apply realistic volume-spread relationship"""
        if avg_volume <= 0:
            return base_spread
        
        # Higher volume = tighter spreads (logarithmic relationship)
        volume_ratio = volume / avg_volume
        volume_factor = 1.0 / (1.0 + 0.15 * np.log(1 + volume_ratio))
        
        # Keep within reasonable bounds
        volume_factor = max(0.7, min(1.3, volume_factor))
        
        return base_spread * volume_factor
    
    def _get_average_volume(self, symbol: str) -> int:
        """Get average volume for a symbol (helper method)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('averageVolume', 1000000)
        except:
            return 1000000  # Default fallback

    
    def is_market_open(self) -> bool:
        """Check if markets are actually open (simplified for demo)"""
        now = datetime.now()
        
        # For demo purposes, assume markets are always open
        # In production, this would check actual market hours
        return True  # Always open for demo
    
    async def get_ultra_realistic_data(self):
        """Get the most realistic market data possible"""
        ultra_real_ticks = []
        
        is_market_open = self.is_market_open()
        current_time = time.time()
        
        for symbol in self.symbols:
            try:
                # Get comprehensive real data
                ticker = yf.Ticker(symbol)
                
                # Real-time quote
                recent_data = ticker.history(period="2d", interval="1m")
                if recent_data.empty:
                    continue
                
                # Get latest bar
                latest = recent_data.iloc[-1]
                
                # Get detailed info
                info = ticker.info
                
                # Calculate REAL metrics
                current_price = float(latest['Close'])
                day_open = float(recent_data.iloc[0]['Open']) if len(recent_data) > 1 else current_price
                day_change = (current_price - day_open) / day_open if day_open > 0 else 0
                
                # Real bid/ask with realistic spreads
                base_spread_pct = 0.001  # 0.1% base spread
                
                # Adjust spread based on market cap
                market_cap = info.get('marketCap', 1e12)
                volatility = self._calculate_real_volatility(recent_data)
                
                # Larger companies = tighter spreads
                market_cap_factor = max(0.5, min(2.0, 1e12 / market_cap))
                volatility_factor = max(0.8, min(2.0, volatility * 20))
                
                real_spread_pct = base_spread_pct * market_cap_factor * volatility_factor
                real_spread_dollars = current_price * real_spread_pct
                

                real_spread_dollars = max(real_spread_dollars, 0.10)  # Min 10 cents
                real_spread_dollars = min(real_spread_dollars, 5.00)   # Max $5

                # ENHANCED: Apply time-of-day effects and volume dynamics
                time_factors = self._get_time_of_day_factors(datetime.fromtimestamp(current_time))
                enhanced_spread = real_spread_dollars * time_factors['spread_factor']

                # Apply volume-spread dynamics
                avg_volume = info.get('averageVolume', 1000000)
                current_volume = int(latest['Volume'])
                final_spread_dollars = self._apply_volume_spread_dynamics(
                    enhanced_spread, current_volume, avg_volume
                )

                # Use final_spread_dollars instead of real_spread_dollars for the rest
                real_spread_dollars = final_spread_dollars
                
                # Real bid/ask
                half_spread = real_spread_dollars / 2
                real_bid = current_price - half_spread
                real_ask = current_price + half_spread
                
                # Liquidity assessment
                avg_volume = info.get('averageVolume', 1000000)
                if avg_volume > 10_000_000:
                    liquidity_tier = "high"
                    base_size = 2000
                elif avg_volume > 1_000_000:
                    liquidity_tier = "medium" 
                    base_size = 1000
                else:
                    liquidity_tier = "low"
                    base_size = 500
                
                print(f"📊 ULTRA-REAL {symbol}: ${current_price:.2f} "
                      f"spread:${real_spread_dollars:.3f} "
                      f"change:{day_change:.2%} "
                      f"liquidity:{liquidity_tier}")
                
                # Create ultra-realistic ticks for each venue
                for i, (venue, venue_info) in enumerate(self.real_venues.items()):
                    # Each venue has slightly different characteristics
                    venue_price_adj = np.random.uniform(-0.0002, 0.0002)  # ±0.02%
                    venue_spread_adj = np.random.uniform(0.95, 1.05)      # ±5% spread variation
                    
                    venue_bid = real_bid * (1 + venue_price_adj)
                    venue_ask = real_ask * (1 + venue_price_adj) * venue_spread_adj
                    
                    # Ensure ask > bid
                    if venue_ask <= venue_bid:
                        venue_ask = venue_bid * 1.0002
                    
                    # Volume varies by venue
                    venue_volume_factor = {
                        'NYSE': 0.3, 'NASDAQ': 0.25, 'ARCA': 0.2, 'IEX': 0.15, 'CBOE': 0.1
                    }
                    venue_volume = int(latest['Volume'] * venue_volume_factor.get(venue, 0.2))
                    
                    # Create ultra-realistic tick
                    tick = RealMarketTick(
                        symbol=symbol,
                        venue=venue,
                        timestamp=current_time + (i * 0.001),
                        bid_price=round(venue_bid, 2),
                        ask_price=round(venue_ask, 2),
                        bid_size=np.random.randint(base_size//2, base_size*2),
                        ask_size=np.random.randint(base_size//2, base_size*2),
                        last_price=current_price,
                        volume=venue_volume,
                        real_spread=venue_ask - venue_bid,
                        market_cap=market_cap,
                        day_change=day_change,
                        volatility=volatility,
                        is_market_hours=is_market_open,
                        exchange_status="open" if is_market_open else "closed",
                        liquidity_tier=liquidity_tier
                    )
                    
                    ultra_real_ticks.append(tick)
                    self.current_prices[symbol] = current_price
                
            except Exception as e:
                print(f"❌ Error fetching ultra-real data for {symbol}: {e}")
        
        return ultra_real_ticks
    
    def _calculate_real_volatility(self, recent_data):
        """Calculate real volatility from recent price action"""
        if len(recent_data) < 10:
            return 0.02
        
        # Use last 20 bars for volatility
        returns = recent_data['Close'].tail(20).pct_change().dropna()
        if len(returns) == 0:
            return 0.02
        
        # Annualized volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        return float(annual_vol)
    
    def _detect_ultra_realistic_arbitrage(self, ticks):
        """Detect arbitrage with more opportunities for demo"""
        # Group by symbol
        by_symbol = {}
        for tick in ticks:
            if tick.symbol not in by_symbol:
                by_symbol[tick.symbol] = []
            by_symbol[tick.symbol].append(tick)
        
        opportunities_found = 0
        
        for symbol, symbol_ticks in by_symbol.items():
            if len(symbol_ticks) < 2:
                continue
            
            for buy_tick in symbol_ticks:
                for sell_tick in symbol_ticks:
                    if buy_tick.venue == sell_tick.venue:
                        continue
                    
                    # REALISTIC but demo-friendly arbitrage calculation
                    gross_profit = sell_tick.bid_price - buy_tick.ask_price
                    
                    # Account for REAL fees
                    buy_fees = buy_tick.ask_price * self.real_venues[buy_tick.venue]['maker_fee']
                    sell_fees = sell_tick.bid_price * self.real_venues[sell_tick.venue]['maker_fee']
                    buy_rebates = buy_tick.ask_price * self.real_venues[buy_tick.venue]['rebate']
                    sell_rebates = sell_tick.bid_price * self.real_venues[sell_tick.venue]['rebate']
                    
                    net_fees = buy_fees + sell_fees - buy_rebates - sell_rebates
                    net_profit = gross_profit - net_fees
                    
                    # More lenient for demo but still realistic
                    if net_profit > 0.03:  # 3 cents after all costs
                        
                        # Reduced competition for demo
                        if np.random.random() < 0.70:  # 70% get taken (was 95%)
                            continue
                        
                        # Size constraints
                        max_size = min(buy_tick.ask_size, sell_tick.bid_size, 500)
                        
                        opportunity = {
                            'symbol': symbol,
                            'timestamp': time.time(),
                            'buy_venue': buy_tick.venue,
                            'sell_venue': sell_tick.venue,
                            'buy_price': buy_tick.ask_price,
                            'sell_price': sell_tick.bid_price,
                            'gross_profit_per_share': gross_profit,
                            'net_profit_per_share': net_profit,
                            'profit_per_share': net_profit,  # For compatibility
                            'max_size': max_size,
                            'buy_fees': buy_fees,
                            'sell_fees': sell_fees,
                            'net_fees': net_fees,
                            'market_hours': buy_tick.is_market_hours,
                            'liquidity_score': (buy_tick.bid_size + sell_tick.ask_size) / 2
                        }
                        
                        self.arbitrage_opportunities.append(opportunity)
                        opportunities_found += 1
                        
                        print(f"🎯 ULTRA-REAL ARBITRAGE: {symbol} "
                              f"buy@{buy_tick.venue}:{buy_tick.ask_price:.2f} "
                              f"sell@{sell_tick.venue}:{sell_tick.bid_price:.2f} "
                              f"net_profit:${net_profit:.3f}")
        
        if opportunities_found == 0:
            print("📊 No arbitrage opportunities found this round")
    
    async def generate_market_data_stream(self, duration_seconds=60):
        """Enhanced stream with symbol-specific update frequencies"""
        print(f"🚀 Starting ENHANCED market data stream for {duration_seconds}s")
        print(f"📊 Mode: {self.mode} | Target: {self.target_ticks_per_minute} ticks/min")
        
        end_time = time.time() + duration_seconds
        last_updates = {symbol: 0 for symbol in self.symbols}
        tick_count_by_symbol = {symbol: 0 for symbol in self.symbols}
        
        while time.time() < end_time:
            try:
                current_time = time.time()
                
                # Check which symbols need updates based on their specific frequencies
                symbols_to_update = []
                for symbol in self.symbols:
                    update_interval = self.enhanced_tick_gen.get_update_interval(symbol)
                    if current_time - last_updates[symbol] >= update_interval:
                        symbols_to_update.append(symbol)
                        last_updates[symbol] = current_time
                
                if symbols_to_update:
                    current_time_str = datetime.now().strftime('%H:%M:%S')
                    print(f"📡 [{current_time_str}] Updating {len(symbols_to_update)} symbols: {symbols_to_update}")
                    
                    # Generate ticks for selected symbols
                    ultra_real_ticks = await self.get_ultra_realistic_data_for_symbols(symbols_to_update)
                    
                    if ultra_real_ticks:
                        # Detect arbitrage opportunities
                        self._detect_ultra_realistic_arbitrage(ultra_real_ticks)
                        
                        # Yield ticks with realistic timing
                        for tick in ultra_real_ticks:
                            self.tick_count += 1
                            tick_count_by_symbol[tick.symbol] += 1
                            yield tick
                            
                            # Small delay between venues
                            await asyncio.sleep(0.01)
                
                # Dynamic sleep based on next update needed
                next_update_times = []
                for symbol in self.symbols:
                    interval = self.enhanced_tick_gen.get_update_interval(symbol)
                    next_update = last_updates[symbol] + interval
                    next_update_times.append(next_update)
                
                if next_update_times:
                    sleep_time = min(max(min(next_update_times) - current_time, 0.1), 2.0)
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in enhanced stream: {e}")
                await asyncio.sleep(1)
        
        # Final statistics
        total_ticks = sum(tick_count_by_symbol.values())
        actual_rate = total_ticks / (duration_seconds / 60) if duration_seconds > 0 else 0
        
        print(f"🏁 Enhanced stream complete!")
        print(f"📊 Total ticks: {total_ticks} | Target: {self.target_ticks_per_minute * (duration_seconds/60):.0f}")
        print(f"📈 Actual rate: {actual_rate:.1f} ticks/min | Target: {self.target_ticks_per_minute}")
        print(f"🎯 Rate efficiency: {(actual_rate/self.target_ticks_per_minute)*100:.1f}%")
        print(f"💎 Arbitrage opportunities: {len(self.arbitrage_opportunities)}")
        
        # Show per-symbol breakdown
        print("📋 Per-symbol tick counts:")
        for symbol, count in sorted(tick_count_by_symbol.items(), key=lambda x: x[1], reverse=True):
            multiplier = self.enhanced_tick_gen.tick_multipliers.get(symbol, 3)
            print(f"   {symbol}: {count} ticks (priority: {multiplier}x)")

    async def get_ultra_realistic_data_for_symbols(self, symbols_to_update):
        """Get realistic data for specific symbols only - ULTRA-REAL FORMAT"""
        ultra_real_ticks = []
        current_time = time.time()
        is_market_open = self.is_market_open()
        
        for symbol in symbols_to_update:
            try:
                # Get comprehensive real data
                ticker = yf.Ticker(symbol)
                
                # Try to get recent data
                recent_data = ticker.history(period="2d", interval="1m")
                if recent_data.empty:
                    recent_data = ticker.history(period="5d", interval="1d")
                
                if recent_data.empty:
                    continue
                    
                # Get latest bar
                latest = recent_data.iloc[-1]
                
                # Get detailed info with fallback
                try:
                    info = ticker.info
                except:
                    info = {'marketCap': 1e12, 'averageVolume': 5000000}
                
                # Calculate REAL metrics
                current_price = float(latest['Close'])
                day_open = float(recent_data.iloc[0]['Open']) if len(recent_data) > 1 else current_price
                day_change = (current_price - day_open) / day_open if day_open > 0 else 0
                
                # Real bid/ask with realistic spreads
                base_spread_pct = 0.001  # 0.1% base spread
                
                # Adjust spread based on market cap
                market_cap = info.get('marketCap', 1e12)
                volatility = self._calculate_real_volatility(recent_data)
                
                # Larger companies = tighter spreads
                market_cap_factor = max(0.5, min(2.0, 1e12 / market_cap))
                volatility_factor = max(0.8, min(2.0, volatility * 20))
                
                real_spread_pct = base_spread_pct * market_cap_factor * volatility_factor
                real_spread_dollars = current_price * real_spread_pct
                real_spread_dollars = max(real_spread_dollars, 0.01)
                real_spread_dollars = min(real_spread_dollars, 2.00)
                
                # Apply time-of-day effects and volume dynamics
                time_factors = self._get_time_of_day_factors(datetime.fromtimestamp(current_time))
                enhanced_spread = real_spread_dollars * time_factors['spread_factor']

                # Apply volume-spread dynamics
                avg_volume = info.get('averageVolume', 1000000)
                current_volume = int(latest['Volume']) if 'Volume' in latest.index else avg_volume
                final_spread_dollars = self._apply_volume_spread_dynamics(
                    enhanced_spread, current_volume, avg_volume
                )
                
                # Real bid/ask
                half_spread = final_spread_dollars / 2
                real_bid = current_price - half_spread
                real_ask = current_price + half_spread
                
                # Liquidity assessment
                if avg_volume > 10_000_000:
                    liquidity_tier = "high"
                    base_size = 2000
                elif avg_volume > 1_000_000:
                    liquidity_tier = "medium" 
                    base_size = 1000
                else:
                    liquidity_tier = "low"
                    base_size = 500
                
                # ✅ KEEP THE ULTRA-REAL FORMAT:
                print(f"📊 REAL {symbol}: ${current_price:.2f} "
                    f"spread:${final_spread_dollars:.3f} "
                    f"change:{day_change:.2%} "
                    f"liquidity:{liquidity_tier}")
                
                # Create ultra-realistic ticks for each venue
                for i, (venue, venue_info) in enumerate(self.real_venues.items()):
                    # Each venue has slightly different characteristics
                    venue_price_adj = np.random.uniform(-0.0002, 0.0002)  # ±0.02%
                    venue_spread_adj = np.random.uniform(0.95, 1.05)      # ±5% spread variation
                    
                    venue_bid = real_bid * (1 + venue_price_adj)
                    venue_ask = real_ask * (1 + venue_price_adj) * venue_spread_adj
                    
                    # Ensure ask > bid
                    if venue_ask <= venue_bid:
                        venue_ask = venue_bid * 1.0002
                    
                    # Volume varies by venue
                    venue_volume_factor = {
                        'NYSE': 0.3, 'NASDAQ': 0.25, 'ARCA': 0.2, 'IEX': 0.15, 'CBOE': 0.1
                    }
                    venue_volume = int(current_volume * venue_volume_factor.get(venue, 0.2))
                    
                    # Create ultra-realistic tick
                    tick = RealMarketTick(
                        symbol=symbol,
                        venue=venue,
                        timestamp=current_time + (i * 0.001),
                        bid_price=round(venue_bid, 2),
                        ask_price=round(venue_ask, 2),
                        bid_size=np.random.randint(base_size//2, base_size*2),
                        ask_size=np.random.randint(base_size//2, base_size*2),
                        last_price=current_price,
                        volume=venue_volume,
                        real_spread=venue_ask - venue_bid,
                        market_cap=market_cap,
                        day_change=day_change,
                        volatility=volatility,
                        is_market_hours=is_market_open,
                        exchange_status="open" if is_market_open else "closed",
                        liquidity_tier=liquidity_tier
                    )
                    
                    ultra_real_ticks.append(tick)
                    self.current_prices[symbol] = current_price
                    
            except Exception as e:
                print(f"❌ Error fetching ultra-real data for {symbol}: {e}")
        
        return ultra_real_ticks
        
    async def initialize_historical_calibration(self):
            """Initialize with real market analysis"""
            print("📊 Performing real market analysis...")
            
            for symbol in self.symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    
                    # Get historical data for calibration
                    hist = ticker.history(period="1mo", interval="1d")
                    if not hist.empty:
                        volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                        avg_volume = hist['Volume'].mean()
                        
                        print(f"✅ {symbol}: {volatility:.1%} annual volatility, "
                            f"{avg_volume:,.0f} avg daily volume")
                    
                except Exception as e:
                    print(f"⚠️  Could not calibrate {symbol}: {e}")
            
            print("✅ Real market calibration complete")
    
    def get_performance_metrics(self):
        """Ultra-realistic performance metrics"""
        return {
            'data_source': 'ULTRA_REALISTIC_LIVE',
            'total_ticks': self.tick_count,
            'real_arbitrage_opportunities': len(self.arbitrage_opportunities),
            'current_prices': dict(self.current_prices),
            'market_hours': self.is_market_open(),
            'last_update': time.time(),
            'venues_monitored': list(self.real_venues.keys())
        }