#!/usr/bin/env python3
"""
Advanced Technical Indicators Engine for HFT
Generates sophisticated features for ML models - NO external APIs needed!
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
# import talib  # Optional - using custom implementations instead
# from scipy import stats  # Not needed - using numpy instead
import warnings
warnings.filterwarnings('ignore')

@dataclass
class IndicatorResult:
    """Structured output for technical indicators"""
    symbol: str
    timestamp: float
    indicators: Dict[str, float]
    microstructure_features: Dict[str, float]
    market_regime: str
    volatility_regime: str
    liquidity_score: float

class AdvancedTechnicalEngine:
    """
    Production-ready technical analysis engine for HFT
    Generates 50+ features from price/volume data
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'short': 20,
            'medium': 50, 
            'long': 200,
            'microstructure': 10
        }
        
        # Rolling windows for different timeframes
        self.price_windows = {
            symbol: {
                'prices': deque(maxlen=self.lookback_periods['long']),
                'volumes': deque(maxlen=self.lookback_periods['long']),
                'high': deque(maxlen=self.lookback_periods['long']),
                'low': deque(maxlen=self.lookback_periods['long']),
                'spreads': deque(maxlen=self.lookback_periods['microstructure']),
                'trade_sizes': deque(maxlen=self.lookback_periods['microstructure'])
            } for symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Add more as needed
        }
        
        print("🧮 Advanced Technical Indicators Engine initialized")
        print(f"📊 Generating 50+ features for ML models")

    def update_data(self, symbol: str, price: float, volume: int, 
                   high: float = None, low: float = None, 
                   bid: float = None, ask: float = None):
        """Update rolling windows with new market data"""
        if symbol not in self.price_windows:
            self.price_windows[symbol] = {
                'prices': deque(maxlen=self.lookback_periods['long']),
                'volumes': deque(maxlen=self.lookback_periods['long']),
                'high': deque(maxlen=self.lookback_periods['long']),
                'low': deque(maxlen=self.lookback_periods['long']),
                'spreads': deque(maxlen=self.lookback_periods['microstructure']),
                'trade_sizes': deque(maxlen=self.lookback_periods['microstructure'])
            }
        
        windows = self.price_windows[symbol]
        windows['prices'].append(price)
        windows['volumes'].append(volume)
        windows['high'].append(high or price)
        windows['low'].append(low or price)
        
        if bid and ask:
            spread = (ask - bid) / ((ask + bid) / 2) * 10000  # bps
            windows['spreads'].append(spread)
        
        if volume > 0:
            windows['trade_sizes'].append(volume)

    def calculate_all_indicators(self, symbol: str, timestamp: float) -> IndicatorResult:
        """Calculate comprehensive technical indicators suite"""
        
        if symbol not in self.price_windows:
            return self._empty_result(symbol, timestamp)
            
        windows = self.price_windows[symbol]
        
        if len(windows['prices']) < self.lookback_periods['short']:
            return self._empty_result(symbol, timestamp)
        
        # Convert to arrays for calculation
        prices = np.array(windows['prices'])
        volumes = np.array(windows['volumes'])
        highs = np.array(windows['high'])
        lows = np.array(windows['low'])
        
        # 1. MOMENTUM INDICATORS
        momentum_features = self._calculate_momentum_indicators(prices)
        
        # 2. VOLATILITY INDICATORS  
        volatility_features = self._calculate_volatility_indicators(prices, highs, lows)
        
        # 3. VOLUME INDICATORS
        volume_features = self._calculate_volume_indicators(prices, volumes)
        
        # 4. MEAN REVERSION INDICATORS
        mean_reversion_features = self._calculate_mean_reversion_indicators(prices)
        
        # 5. MICROSTRUCTURE FEATURES
        microstructure_features = self._calculate_microstructure_features(symbol)
        
        # 6. REGIME DETECTION
        market_regime = self._detect_market_regime(prices, volumes)
        volatility_regime = self._detect_volatility_regime(prices)
        
        # 7. LIQUIDITY ASSESSMENT
        liquidity_score = self._calculate_liquidity_score(symbol)
        
        # Combine all features
        all_indicators = {
            **momentum_features,
            **volatility_features, 
            **volume_features,
            **mean_reversion_features
        }
        
        return IndicatorResult(
            symbol=symbol,
            timestamp=timestamp,
            indicators=all_indicators,
            microstructure_features=microstructure_features,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
            liquidity_score=liquidity_score
        )

    def _calculate_momentum_indicators(self, prices: np.array) -> Dict[str, float]:
        """Calculate momentum-based indicators"""
        features = {}
        
        try:
            # RSI variations
            features['rsi_14'] = self._calculate_rsi(prices, 14)
            features['rsi_21'] = self._calculate_rsi(prices, 21)
            features['rsi_slope'] = self._calculate_slope(prices[-10:]) if len(prices) >= 10 else 0
            
            # MACD family
            macd, macd_signal, macd_hist = self._calculate_macd(prices)
            features['macd'] = macd
            features['macd_signal'] = macd_signal  
            features['macd_histogram'] = macd_hist
            features['macd_divergence'] = macd - macd_signal
            
            # Rate of Change variations
            features['roc_1'] = (prices[-1] / prices[-2] - 1) * 100 if len(prices) >= 2 else 0
            features['roc_5'] = (prices[-1] / prices[-6] - 1) * 100 if len(prices) >= 6 else 0
            features['roc_10'] = (prices[-1] / prices[-11] - 1) * 100 if len(prices) >= 11 else 0
            
            # Momentum oscillators
            features['momentum_10'] = prices[-1] / prices[-11] if len(prices) >= 11 else 1
            features['momentum_20'] = prices[-1] / prices[-21] if len(prices) >= 21 else 1
            
            # Williams %R
            features['williams_r'] = self._calculate_williams_r(prices[-14:], prices[-14:], prices[-14:]) if len(prices) >= 14 else -50
            
        except Exception as e:
            print(f"⚠️ Momentum calculation error: {e}")
            features = {k: 0.0 for k in ['rsi_14', 'rsi_21', 'rsi_slope', 'macd', 'macd_signal', 
                                       'macd_histogram', 'macd_divergence', 'roc_1', 'roc_5', 'roc_10',
                                       'momentum_10', 'momentum_20', 'williams_r']}
        
        return features

    def _calculate_volatility_indicators(self, prices: np.array, highs: np.array, lows: np.array) -> Dict[str, float]:
        """Calculate volatility-based indicators"""
        features = {}
        
        try:
            # Bollinger Bands
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            std_20 = np.std(prices[-20:]) if len(prices) >= 20 else 0
            features['bb_upper'] = sma_20 + (2 * std_20)
            features['bb_lower'] = sma_20 - (2 * std_20)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma_20 * 100
            features['bb_position'] = (prices[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']) if features['bb_upper'] != features['bb_lower'] else 0.5
            
            # ATR (Average True Range)
            features['atr_14'] = self._calculate_atr(highs, lows, prices, 14) if len(prices) >= 14 else 0
            features['atr_21'] = self._calculate_atr(highs, lows, prices, 21) if len(prices) >= 21 else 0
            
            # Volatility measures
            features['volatility_10'] = np.std(prices[-10:]) / np.mean(prices[-10:]) * 100 if len(prices) >= 10 else 0
            features['volatility_20'] = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100 if len(prices) >= 20 else 0
            
            # Keltner Channels
            ema_20 = self._calculate_ema(prices, 20)
            atr_20 = features['atr_21']
            features['keltner_upper'] = ema_20 + (2 * atr_20)
            features['keltner_lower'] = ema_20 - (2 * atr_20)
            features['keltner_position'] = (prices[-1] - features['keltner_lower']) / (features['keltner_upper'] - features['keltner_lower']) if features['keltner_upper'] != features['keltner_lower'] else 0.5
            
            # Volatility breakouts
            features['volatility_breakout'] = 1 if features['volatility_10'] > features['volatility_20'] * 1.5 else 0
            
        except Exception as e:
            print(f"⚠️ Volatility calculation error: {e}")
            features = {k: 0.0 for k in ['bb_upper', 'bb_lower', 'bb_width', 'bb_position', 
                                       'atr_14', 'atr_21', 'volatility_10', 'volatility_20',
                                       'keltner_upper', 'keltner_lower', 'keltner_position', 'volatility_breakout']}
        
        return features

    def _calculate_volume_indicators(self, prices: np.array, volumes: np.array) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        features = {}
        
        try:
            # Volume moving averages
            features['volume_sma_10'] = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
            features['volume_sma_20'] = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            
            # Volume ratios
            features['volume_ratio_short'] = volumes[-1] / features['volume_sma_10'] if features['volume_sma_10'] > 0 else 1
            features['volume_ratio_long'] = volumes[-1] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1
            
            # On-Balance Volume
            features['obv'] = self._calculate_obv(prices, volumes)
            
            # Volume Price Trend
            features['vpt'] = self._calculate_vpt(prices, volumes)
            
            # Price Volume Trend
            features['pvt'] = self._calculate_pvt(prices, volumes)
            
            # Volume breakouts
            features['volume_breakout'] = 1 if volumes[-1] > features['volume_sma_20'] * 2 else 0
            
            # VWAP deviation
            vwap = self._calculate_vwap(prices[-20:], volumes[-20:]) if len(prices) >= 20 else prices[-1]
            features['vwap_deviation'] = (prices[-1] - vwap) / vwap * 100
            
        except Exception as e:
            print(f"⚠️ Volume calculation error: {e}")
            features = {k: 0.0 for k in ['volume_sma_10', 'volume_sma_20', 'volume_ratio_short', 
                                       'volume_ratio_long', 'obv', 'vpt', 'pvt', 'volume_breakout', 'vwap_deviation']}
        
        return features

    def _calculate_mean_reversion_indicators(self, prices: np.array) -> Dict[str, float]:
        """Calculate mean reversion indicators"""
        features = {}
        
        try:
            # Moving average deviations
            sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            features['deviation_sma_10'] = (prices[-1] - sma_10) / sma_10 * 100
            features['deviation_sma_20'] = (prices[-1] - sma_20) / sma_20 * 100
            features['deviation_sma_50'] = (prices[-1] - sma_50) / sma_50 * 100
            
            # EMA deviations  
            ema_12 = self._calculate_ema(prices, 12)
            ema_26 = self._calculate_ema(prices, 26)
            features['deviation_ema_12'] = (prices[-1] - ema_12) / ema_12 * 100
            features['deviation_ema_26'] = (prices[-1] - ema_26) / ema_26 * 100
            
            # Mean reversion strength
            z_score_10 = (prices[-1] - sma_10) / np.std(prices[-10:]) if len(prices) >= 10 and np.std(prices[-10:]) > 0 else 0
            z_score_20 = (prices[-1] - sma_20) / np.std(prices[-20:]) if len(prices) >= 20 and np.std(prices[-20:]) > 0 else 0
            
            features['z_score_10'] = z_score_10
            features['z_score_20'] = z_score_20
            
            # Mean reversion signals
            features['mean_reversion_signal'] = 1 if abs(z_score_20) > 2 else 0
            features['overextended'] = 1 if abs(features['deviation_sma_20']) > 5 else 0
            
        except Exception as e:
            print(f"⚠️ Mean reversion calculation error: {e}")
            features = {k: 0.0 for k in ['deviation_sma_10', 'deviation_sma_20', 'deviation_sma_50',
                                       'deviation_ema_12', 'deviation_ema_26', 'z_score_10', 'z_score_20',
                                       'mean_reversion_signal', 'overextended']}
        
        return features

    def _calculate_microstructure_features(self, symbol: str) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {}
        
        try:
            windows = self.price_windows[symbol]
            
            # Spread analysis
            spreads = list(windows['spreads'])
            if len(spreads) >= 5:
                features['avg_spread'] = np.mean(spreads)
                features['spread_volatility'] = np.std(spreads)
                features['spread_trend'] = self._calculate_slope(spreads[-5:])
            else:
                features.update({'avg_spread': 0, 'spread_volatility': 0, 'spread_trend': 0})
            
            # Trade size analysis
            trade_sizes = list(windows['trade_sizes'])
            if len(trade_sizes) >= 5:
                features['avg_trade_size'] = np.mean(trade_sizes)
                features['trade_size_volatility'] = np.std(trade_sizes)
                features['large_trade_ratio'] = sum(1 for size in trade_sizes[-5:] if size > features['avg_trade_size'] * 2) / 5
            else:
                features.update({'avg_trade_size': 0, 'trade_size_volatility': 0, 'large_trade_ratio': 0})
            
            # Market impact proxies
            prices = list(windows['prices'])
            if len(prices) >= 5:
                returns = np.diff(prices[-5:]) / prices[-5:-1]
                features['return_volatility'] = np.std(returns) if len(returns) > 0 else 0
                features['return_autocorr'] = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 2 else 0
            else:
                features.update({'return_volatility': 0, 'return_autocorr': 0})
                
        except Exception as e:
            print(f"⚠️ Microstructure calculation error: {e}")
            features = {k: 0.0 for k in ['avg_spread', 'spread_volatility', 'spread_trend',
                                       'avg_trade_size', 'trade_size_volatility', 'large_trade_ratio',
                                       'return_volatility', 'return_autocorr']}
        
        return features

    def _detect_market_regime(self, prices: np.array, volumes: np.array) -> str:
        """Detect current market regime"""
        try:
            if len(prices) < 20:
                return 'insufficient_data'
            
            # Trend detection
            slope = self._calculate_slope(prices[-20:])
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            volume_avg = np.mean(volumes[-10:])
            volume_current = volumes[-1]
            
            if slope > 0.002 and volatility < 0.02:
                return 'trending_up_low_vol'
            elif slope > 0.002 and volatility >= 0.02:
                return 'trending_up_high_vol'  
            elif slope < -0.002 and volatility < 0.02:
                return 'trending_down_low_vol'
            elif slope < -0.002 and volatility >= 0.02:
                return 'trending_down_high_vol'
            elif volatility >= 0.03:
                return 'high_volatility'
            elif volume_current > volume_avg * 2:
                return 'high_volume'
            else:
                return 'sideways'
                
        except Exception:
            return 'unknown'

    def _detect_volatility_regime(self, prices: np.array) -> str:
        """Detect volatility regime"""
        try:
            if len(prices) < 20:
                return 'insufficient_data'
            
            current_vol = np.std(prices[-10:]) / np.mean(prices[-10:])
            historical_vol = np.std(prices[-20:-10]) / np.mean(prices[-20:-10])
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            
            if vol_ratio > 1.5:
                return 'vol_expanding'
            elif vol_ratio < 0.7:
                return 'vol_contracting' 
            else:
                return 'vol_stable'
                
        except Exception:
            return 'unknown'

    def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score (0-100)"""
        try:
            windows = self.price_windows[symbol]
            
            # Factors: spread, volume, price volatility
            spreads = list(windows['spreads'])
            volumes = list(windows['volumes'])
            prices = list(windows['prices'])
            
            if len(spreads) < 3 or len(volumes) < 3:
                return 50.0  # Neutral score
            
            # Lower spread = higher liquidity
            avg_spread = np.mean(spreads)
            spread_score = max(0, 100 - avg_spread * 10)  # Assume spread in bps
            
            # Higher volume = higher liquidity  
            avg_volume = np.mean(volumes)
            volume_score = min(100, avg_volume / 10000 * 100)  # Normalize by 10k
            
            # Lower volatility = higher liquidity
            volatility = np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 else 0
            volatility_score = max(0, 100 - volatility * 1000)
            
            # Weighted average
            liquidity_score = (spread_score * 0.4 + volume_score * 0.4 + volatility_score * 0.2)
            
            return max(0, min(100, liquidity_score))
            
        except Exception:
            return 50.0

    # Helper methods
    def _calculate_rsi(self, prices: np.array, period: int) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception:
            return 50.0

    def _calculate_macd(self, prices: np.array, fast=12, slow=26, signal=9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        try:
            if len(prices) < slow + signal:
                return 0.0, 0.0, 0.0
            
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            
            # Simple approximation for signal line
            macd_signal = macd_line * 0.9  # Simplified
            macd_histogram = macd_line - macd_signal
            
            return macd_line, macd_signal, macd_histogram
        except Exception:
            return 0.0, 0.0, 0.0

    def _calculate_ema(self, prices: np.array, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) == 0:
                return 0.0
            if len(prices) == 1:
                return prices[0]
                
            alpha = 2 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
                
            return ema
        except Exception:
            return prices[-1] if len(prices) > 0 else 0.0

    def _calculate_atr(self, highs: np.array, lows: np.array, closes: np.array, period: int) -> float:
        """Calculate Average True Range"""
        try:
            if len(highs) < period:
                return 0.0
            
            true_ranges = []
            for i in range(1, len(highs)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1]) 
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            return np.mean(true_ranges[-period:]) if len(true_ranges) >= period else 0.0
        except Exception:
            return 0.0

    def _calculate_williams_r(self, highs: np.array, lows: np.array, closes: np.array) -> float:
        """Calculate Williams %R"""
        try:
            if len(highs) == 0:
                return -50.0
                
            highest_high = np.max(highs)
            lowest_low = np.min(lows)
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                return -50.0
                
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
            return williams_r
        except Exception:
            return -50.0

    def _calculate_obv(self, prices: np.array, volumes: np.array) -> float:
        """Calculate On-Balance Volume"""
        try:
            if len(prices) < 2:
                return 0.0
                
            obv = 0
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv += volumes[i]
                elif prices[i] < prices[i-1]:
                    obv -= volumes[i]
                    
            return obv
        except Exception:
            return 0.0

    def _calculate_vpt(self, prices: np.array, volumes: np.array) -> float:
        """Calculate Volume Price Trend"""
        try:
            if len(prices) < 2:
                return 0.0
                
            vpt = 0
            for i in range(1, len(prices)):
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                vpt += volumes[i] * price_change
                
            return vpt
        except Exception:
            return 0.0

    def _calculate_pvt(self, prices: np.array, volumes: np.array) -> float:
        """Calculate Price Volume Trend"""
        return self._calculate_vpt(prices, volumes)  # Same calculation

    def _calculate_vwap(self, prices: np.array, volumes: np.array) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            if len(prices) == 0 or len(volumes) == 0:
                return 0.0
                
            total_pv = np.sum(prices * volumes)
            total_volume = np.sum(volumes)
            
            return total_pv / total_volume if total_volume > 0 else prices[-1]
        except Exception:
            return prices[-1] if len(prices) > 0 else 0.0

    def _calculate_slope(self, values: np.array) -> float:
        """Calculate linear regression slope"""
        try:
            if len(values) < 2:
                return 0.0
                
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        except Exception:
            return 0.0

    def _empty_result(self, symbol: str, timestamp: float) -> IndicatorResult:
        """Return empty result structure"""
        return IndicatorResult(
            symbol=symbol,
            timestamp=timestamp,
            indicators={},
            microstructure_features={},
            market_regime='insufficient_data',
            volatility_regime='insufficient_data',
            liquidity_score=50.0
        )

    def get_ml_feature_vector(self, symbol: str, timestamp: float) -> np.array:
        """Get feature vector optimized for ML models"""
        result = self.calculate_all_indicators(symbol, timestamp)
        
        # Combine all features into single vector
        feature_vector = []
        
        # Add indicator values
        for key in sorted(result.indicators.keys()):
            feature_vector.append(result.indicators[key])
        
        # Add microstructure features  
        for key in sorted(result.microstructure_features.keys()):
            feature_vector.append(result.microstructure_features[key])
        
        # Add regime encodings
        regime_encoding = {
            'trending_up_low_vol': 1, 'trending_up_high_vol': 2,
            'trending_down_low_vol': 3, 'trending_down_high_vol': 4,
            'high_volatility': 5, 'high_volume': 6, 'sideways': 7
        }
        feature_vector.append(regime_encoding.get(result.market_regime, 0))
        
        vol_regime_encoding = {
            'vol_expanding': 1, 'vol_contracting': 2, 'vol_stable': 3
        }
        feature_vector.append(vol_regime_encoding.get(result.volatility_regime, 0))
        
        # Add liquidity score
        feature_vector.append(result.liquidity_score)
        
        return np.array(feature_vector)

# Integration with existing system
def integrate_technical_indicators(market_data_generator, latency_predictor):
    """Integration function to add technical indicators to existing system"""
    
    technical_engine = AdvancedTechnicalEngine()
    
    # Monkey patch the market data generator
    original_generate_tick = market_data_generator.generate_realistic_tick
    
    def enhanced_generate_tick(*args, **kwargs):
        tick = original_generate_tick(*args, **kwargs)
        
        # Update technical indicators
        technical_engine.update_data(
            symbol=tick.symbol,
            price=tick.mid_price,
            volume=tick.volume,
            high=tick.last_price * 1.001,  # Approximate
            low=tick.last_price * 0.999,   # Approximate  
            bid=tick.bid_price,
            ask=tick.ask_price
        )
        
        # Add technical features to tick
        indicators = technical_engine.calculate_all_indicators(tick.symbol, tick.timestamp)
        tick.technical_indicators = indicators
        tick.ml_features = technical_engine.get_ml_feature_vector(tick.symbol, tick.timestamp)
        
        return tick
    
    market_data_generator.generate_realistic_tick = enhanced_generate_tick
    
    return technical_engine

if __name__ == "__main__":
    # Test the engine
    engine = AdvancedTechnicalEngine()
    
    # Simulate some data
    import time
    test_symbol = "AAPL"
    base_price = 150.0
    
    print("🧪 Testing Advanced Technical Indicators...")
    
    for i in range(50):
        price = base_price + np.random.normal(0, 1)
        volume = int(np.random.exponential(10000))
        
        engine.update_data(
            symbol=test_symbol,
            price=price,
            volume=volume, 
            high=price * 1.002,
            low=price * 0.998,
            bid=price * 0.999,
            ask=price * 1.001
        )
    
    # Get full analysis
    result = engine.calculate_all_indicators(test_symbol, time.time())
    
    print(f"\n📊 Analysis Results for {test_symbol}:")
    print(f"Market Regime: {result.market_regime}")
    print(f"Volatility Regime: {result.volatility_regime}")
    print(f"Liquidity Score: {result.liquidity_score:.1f}")
    print(f"Indicators Generated: {len(result.indicators)}")
    print(f"Microstructure Features: {len(result.microstructure_features)}")
    
    # Show ML feature vector
    ml_features = engine.get_ml_feature_vector(test_symbol, time.time())
    print(f"ML Feature Vector Length: {len(ml_features)}")
    print("✅ Technical Indicators Engine Test Complete!")