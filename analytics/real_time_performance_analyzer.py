#!/usr/bin/env python3
"""
Real-Time Performance Analytics for HFT System
Comprehensive performance tracking and optimization insights - NO external dependencies!
"""

import time
import numpy as np
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import threading
import statistics
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetric(Enum):
    EXECUTION_LATENCY = "execution_latency_us"
    SLIPPAGE = "slippage_bps"
    FILL_RATE = "fill_rate_pct"
    MARKET_IMPACT = "market_impact_bps"
    PROFIT_LOSS = "pnl_bps"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall_bps"
    ARRIVAL_PRICE_PERFORMANCE = "arrival_price_performance_bps"
    VENUE_ROUTING_EFFICIENCY = "venue_routing_efficiency_pct"

@dataclass
class TradeExecution:
    """Single trade execution record"""
    trade_id: str
    timestamp: float
    symbol: str
    venue: str
    side: str  # 'buy' or 'sell'
    quantity: int
    requested_price: float
    executed_price: float
    arrival_price: float  # Price when order decision was made
    execution_latency_us: float
    slippage_bps: float
    market_impact_bps: float
    fees: float
    strategy: str
    predicted_latency_us: Optional[float] = None
    ml_confidence: Optional[float] = None

@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: float
    window_minutes: int
    total_trades: int
    total_volume: float
    total_pnl: float
    metrics: Dict[str, float]
    venue_performance: Dict[str, Dict[str, float]]
    strategy_performance: Dict[str, Dict[str, float]]
    top_insights: List[str]

class RealTimePerformanceAnalyzer:
    """
    Real-time performance analytics engine
    Tracks execution quality, identifies optimization opportunities
    """
    
    def __init__(self, max_trades_history: int = 10000):
        self.max_trades_history = max_trades_history
        
        # Trade execution history
        self.trade_history = deque(maxlen=max_trades_history)
        
        # Real-time metrics (rolling windows)
        self.rolling_metrics = {
            '1min': deque(maxlen=60),   # 60 seconds of data
            '5min': deque(maxlen=300),  # 5 minutes
            '15min': deque(maxlen=900), # 15 minutes  
            '1hour': deque(maxlen=3600) # 1 hour
        }
        
        # Performance by venue
        self.venue_performance = defaultdict(lambda: {
            'trades': deque(maxlen=1000),
            'total_volume': 0,
            'total_pnl': 0,
            'avg_latency': 0,
            'avg_slippage': 0,
            'fill_rate': 100.0
        })
        
        # Performance by strategy
        self.strategy_performance = defaultdict(lambda: {
            'trades': deque(maxlen=1000),
            'total_pnl': 0,
            'win_rate': 0,
            'avg_trade_size': 0,
            'sharpe_ratio': 0
        })
        
        # Benchmarking data
        self.benchmark_metrics = {
            'execution_latency_p50': 800,  # μs
            'execution_latency_p95': 2000,
            'slippage_p50': 1.5,  # bps
            'slippage_p95': 5.0,
            'fill_rate_target': 98.5,  # %
            'market_impact_target': 2.0  # bps
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_latency': 3000,  # μs
            'high_slippage': 10.0,  # bps
            'low_fill_rate': 95.0,  # %
            'high_market_impact': 8.0  # bps
        }
        
        # Analytics callbacks
        self.analytics_callbacks = []
        
        # Background analysis
        self.is_analyzing = False
        self.analysis_thread = None
        
        print("📊 Real-Time Performance Analyzer initialized")
        print(f"📈 Tracking {len(PerformanceMetric)} key performance metrics")

    def record_trade_execution(self, execution: TradeExecution):
        """Record a new trade execution"""
        self.trade_history.append(execution)
        
        # Update venue performance
        venue_stats = self.venue_performance[execution.venue]
        venue_stats['trades'].append(execution)
        venue_stats['total_volume'] += abs(execution.quantity * execution.executed_price)
        venue_stats['total_pnl'] += self._calculate_trade_pnl(execution)
        
        # Update strategy performance
        strategy_stats = self.strategy_performance[execution.strategy]
        strategy_stats['trades'].append(execution)
        strategy_stats['total_pnl'] += self._calculate_trade_pnl(execution)
        
        # Add to rolling windows
        current_time = time.time()
        metric_snapshot = self._create_metric_snapshot(execution, current_time)
        
        for window_name, window_data in self.rolling_metrics.items():
            window_data.append(metric_snapshot)
        
        # Trigger real-time analysis
        self._trigger_real_time_analysis(execution)

    def _calculate_trade_pnl(self, execution: TradeExecution) -> float:
        """Calculate P&L for a single trade"""
        # Simplified P&L calculation
        # In production, this would be more sophisticated with position tracking
        
        if execution.side.lower() == 'buy':
            # For buy orders, positive slippage is bad (paid more)
            pnl = -execution.slippage_bps * execution.quantity * execution.executed_price / 10000
        else:
            # For sell orders, negative slippage is bad (received less)
            pnl = execution.slippage_bps * execution.quantity * execution.executed_price / 10000
        
        # Subtract fees
        pnl -= execution.fees
        
        return pnl

    def _create_metric_snapshot(self, execution: TradeExecution, timestamp: float) -> Dict[str, Any]:
        """Create metrics snapshot for this execution"""
        return {
            'timestamp': timestamp,
            'execution_latency_us': execution.execution_latency_us,
            'slippage_bps': execution.slippage_bps,
            'market_impact_bps': execution.market_impact_bps,
            'venue': execution.venue,
            'strategy': execution.strategy,
            'symbol': execution.symbol,
            'trade_pnl': self._calculate_trade_pnl(execution),
            'volume': abs(execution.quantity * execution.executed_price)
        }

    def get_real_time_metrics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Get real-time performance metrics"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        # Get recent trades
        recent_trades = [
            trade for trade in self.trade_history
            if current_time - trade.timestamp <= window_seconds
        ]
        
        if not recent_trades:
            return {}
        
        # Calculate key metrics
        metrics = {}
        
        # Execution Latency
        latencies = [t.execution_latency_us for t in recent_trades]
        metrics['avg_execution_latency_us'] = statistics.mean(latencies)
        metrics['p50_execution_latency_us'] = statistics.median(latencies)
        metrics['p95_execution_latency_us'] = np.percentile(latencies, 95)
        
        # Slippage
        slippages = [t.slippage_bps for t in recent_trades]
        metrics['avg_slippage_bps'] = statistics.mean(slippages)
        metrics['p50_slippage_bps'] = statistics.median(slippages)
        metrics['p95_slippage_bps'] = np.percentile(slippages, 95)
        
        # Market Impact
        impacts = [t.market_impact_bps for t in recent_trades]
        metrics['avg_market_impact_bps'] = statistics.mean(impacts)
        metrics['p95_market_impact_bps'] = np.percentile(impacts, 95)
        
        # Fill Rate (simplified - assuming all trades filled)
        metrics['fill_rate_pct'] = 100.0
        
        # Volume and Trade Count
        metrics['total_trades'] = len(recent_trades)
        metrics['total_volume'] = sum(abs(t.quantity * t.executed_price) for t in recent_trades)
        metrics['avg_trade_size'] = metrics['total_volume'] / len(recent_trades)
        
        # P&L
        total_pnl = sum(self._calculate_trade_pnl(t) for t in recent_trades)
        metrics['total_pnl'] = total_pnl
        metrics['pnl_per_trade'] = total_pnl / len(recent_trades)
        
        # Implementation Shortfall
        implementation_shortfall = self._calculate_implementation_shortfall(recent_trades)
        metrics['implementation_shortfall_bps'] = implementation_shortfall
        
        # Arrival Price Performance  
        arrival_performance = self._calculate_arrival_price_performance(recent_trades)
        metrics['arrival_price_performance_bps'] = arrival_performance
        
        return metrics

    def _calculate_implementation_shortfall(self, trades: List[TradeExecution]) -> float:
        """Calculate implementation shortfall"""
        if not trades:
            return 0.0
        
        total_shortfall = 0.0
        total_value = 0.0
        
        for trade in trades:
            # Implementation shortfall = (executed_price - arrival_price) / arrival_price
            # For buy orders, positive shortfall is bad; for sell orders, negative shortfall is bad
            
            if trade.side.lower() == 'buy':
                shortfall_per_share = trade.executed_price - trade.arrival_price
            else:
                shortfall_per_share = trade.arrival_price - trade.executed_price
            
            trade_value = abs(trade.quantity * trade.arrival_price)
            shortfall_bps = (shortfall_per_share / trade.arrival_price) * 10000
            
            total_shortfall += shortfall_bps * trade_value
            total_value += trade_value
        
        return total_shortfall / total_value if total_value > 0 else 0.0

    def _calculate_arrival_price_performance(self, trades: List[TradeExecution]) -> float:
        """Calculate arrival price performance"""
        if not trades:
            return 0.0
        
        total_performance = 0.0
        total_value = 0.0
        
        for trade in trades:
            # How well did we do relative to arrival price?
            trade_value = abs(trade.quantity * trade.arrival_price)
            
            if trade.side.lower() == 'buy':
                # For buys, lower execution price is better
                performance = (trade.arrival_price - trade.executed_price) / trade.arrival_price * 10000
            else:
                # For sells, higher execution price is better
                performance = (trade.executed_price - trade.arrival_price) / trade.arrival_price * 10000
            
            total_performance += performance * trade_value
            total_value += trade_value
        
        return total_performance / total_value if total_value > 0 else 0.0

    def get_venue_performance_ranking(self, window_minutes: int = 15) -> List[Tuple[str, Dict[str, float]]]:
        """Get venues ranked by performance"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        venue_rankings = []
        
        for venue, stats in self.venue_performance.items():
            # Get recent trades for this venue
            recent_trades = [
                trade for trade in stats['trades']
                if current_time - trade.timestamp <= window_seconds
            ]
            
            if not recent_trades:
                continue
            
            # Calculate venue-specific metrics
            avg_latency = statistics.mean([t.execution_latency_us for t in recent_trades])
            avg_slippage = statistics.mean([t.slippage_bps for t in recent_trades])
            total_volume = sum(abs(t.quantity * t.executed_price) for t in recent_trades)
            total_pnl = sum(self._calculate_trade_pnl(t) for t in recent_trades)
            
            # Calculate composite score (lower is better)
            latency_score = avg_latency / 1000  # Normalize to ms
            slippage_score = max(0, avg_slippage)  # Only count positive slippage as bad
            
            composite_score = latency_score + slippage_score
            
            venue_performance = {
                'avg_latency_us': avg_latency,
                'avg_slippage_bps': avg_slippage,
                'total_volume': total_volume,
                'total_pnl': total_pnl,
                'trade_count': len(recent_trades),
                'composite_score': composite_score,
                'pnl_per_dollar': total_pnl / total_volume if total_volume > 0 else 0
            }
            
            venue_rankings.append((venue, venue_performance))
        
        # Sort by composite score (lower is better)
        venue_rankings.sort(key=lambda x: x[1]['composite_score'])
        
        return venue_rankings

    def get_strategy_performance_analysis(self, window_minutes: int = 30) -> Dict[str, Dict[str, float]]:
        """Analyze performance by trading strategy"""
        current_time = time.time()
        window_seconds = window_minutes * 60
        
        strategy_analysis = {}
        
        for strategy, stats in self.strategy_performance.items():
            recent_trades = [
                trade for trade in stats['trades']
                if current_time - trade.timestamp <= window_seconds
            ]
            
            if not recent_trades:
                continue
            
            # Calculate strategy metrics
            total_pnl = sum(self._calculate_trade_pnl(t) for t in recent_trades)
            total_volume = sum(abs(t.quantity * t.executed_price) for t in recent_trades)
            
            winning_trades = [t for t in recent_trades if self._calculate_trade_pnl(t) > 0]
            win_rate = len(winning_trades) / len(recent_trades) * 100
            
            # Calculate Sharpe ratio approximation
            trade_returns = [self._calculate_trade_pnl(t) / abs(t.quantity * t.executed_price) 
                           for t in recent_trades]
            avg_return = statistics.mean(trade_returns)
            return_volatility = statistics.stdev(trade_returns) if len(trade_returns) > 1 else 0
            sharpe_ratio = avg_return / return_volatility if return_volatility > 0 else 0
            
            strategy_analysis[strategy] = {
                'total_pnl': total_pnl,
                'trade_count': len(recent_trades),
                'win_rate_pct': win_rate,
                'avg_trade_size': total_volume / len(recent_trades),
                'pnl_per_trade': total_pnl / len(recent_trades),
                'pnl_per_dollar': total_pnl / total_volume if total_volume > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'avg_latency': statistics.mean([t.execution_latency_us for t in recent_trades]),
                'avg_slippage': statistics.mean([t.slippage_bps for t in recent_trades])
            }
        
        return strategy_analysis

    def get_performance_insights(self, window_minutes: int = 10) -> List[str]:
        """Generate actionable performance insights"""
        insights = []
        
        # Get current metrics
        metrics = self.get_real_time_metrics(window_minutes)
        
        if not metrics:
            return ["Insufficient data for analysis"]
        
        # Latency insights
        if metrics.get('avg_execution_latency_us', 0) > self.alert_thresholds['high_latency']:
            insights.append(f"⚠️ High execution latency: {metrics['avg_execution_latency_us']:.0f}μs "
                          f"(target: <{self.alert_thresholds['high_latency']}μs)")
        
        if metrics.get('p95_execution_latency_us', 0) > self.benchmark_metrics['execution_latency_p95']:
            insights.append(f"📈 P95 latency above benchmark: {metrics['p95_execution_latency_us']:.0f}μs "
                          f"vs {self.benchmark_metrics['execution_latency_p95']}μs target")
        
        # Slippage insights
        if metrics.get('avg_slippage_bps', 0) > self.alert_thresholds['high_slippage']:
            insights.append(f"💸 High slippage: {metrics['avg_slippage_bps']:.1f}bps "
                          f"(target: <{self.alert_thresholds['high_slippage']}bps)")
        
        # Implementation shortfall insights
        is_bps = metrics.get('implementation_shortfall_bps', 0)
        if is_bps > 5.0:
            insights.append(f"📉 Implementation shortfall high: {is_bps:.1f}bps - consider order sizing or timing")
        elif is_bps < -2.0:
            insights.append(f"📈 Implementation shortfall negative: {is_bps:.1f}bps - excellent execution timing")
        
        # Venue performance insights
        venue_rankings = self.get_venue_performance_ranking(window_minutes)
        if len(venue_rankings) > 1:
            best_venue = venue_rankings[0]
            worst_venue = venue_rankings[-1]
            
            performance_gap = worst_venue[1]['composite_score'] - best_venue[1]['composite_score']
            if performance_gap > 2.0:
                insights.append(f"🏆 Consider routing more to {best_venue[0]} "
                              f"(outperforming {worst_venue[0]} by {performance_gap:.1f} points)")
        
        # P&L insights
        total_pnl = metrics.get('total_pnl', 0)
        pnl_per_trade = metrics.get('pnl_per_trade', 0)
        
        if total_pnl > 0:
            insights.append(f"💰 Profitable period: +${total_pnl:.2f} "
                          f"(${pnl_per_trade:.2f} per trade)")
        elif total_pnl < -100:
            insights.append(f"🔴 Loss period: ${total_pnl:.2f} - review strategy parameters")
        
        # Volume insights
        trade_count = metrics.get('total_trades', 0)
        if trade_count > 100:
            insights.append(f"📊 High activity: {trade_count} trades - monitor system resources")
        elif trade_count < 5:
            insights.append(f"🐌 Low activity: {trade_count} trades - check market conditions or signals")
        
        # ML prediction accuracy (if available)
        trades_with_predictions = [t for t in self.trade_history 
                                 if t.predicted_latency_us is not None
                                 and time.time() - t.timestamp <= window_minutes * 60]
        
        if len(trades_with_predictions) > 5:
            prediction_errors = [abs(t.execution_latency_us - t.predicted_latency_us) 
                               for t in trades_with_predictions]
            avg_error = statistics.mean(prediction_errors)
            avg_actual = statistics.mean([t.execution_latency_us for t in trades_with_predictions])
            accuracy_pct = max(0, 100 - (avg_error / avg_actual * 100))
            
            if accuracy_pct < 70:
                insights.append(f"🤖 ML prediction accuracy low: {accuracy_pct:.1f}% - retrain models")
            elif accuracy_pct > 90:
                insights.append(f"🎯 ML prediction accuracy excellent: {accuracy_pct:.1f}%")
        
        # If no insights, provide positive feedback
        if not insights:
            insights.append("✅ All metrics within normal ranges - system performing well")
        
        return insights

    def _trigger_real_time_analysis(self, execution: TradeExecution):
        """Trigger real-time analysis callbacks"""
        for callback in self.analytics_callbacks:
            try:
                callback(execution, self.get_real_time_metrics())
            except Exception as e:
                print(f"Analytics callback error: {e}")

    def add_analytics_callback(self, callback: Callable[[TradeExecution, Dict[str, float]], None]):
        """Add callback for real-time analytics"""
        self.analytics_callbacks.append(callback)

    def start_continuous_analysis(self, analysis_interval_seconds: float = 10.0):
        """Start continuous background analysis"""
        if self.is_analyzing:
            print("⚠️ Continuous analysis already running")
            return
        
        self.is_analyzing = True
        
        def analysis_loop():
            while self.is_analyzing:
                try:
                    # Generate insights
                    insights = self.get_performance_insights()
                    
                    if insights:
                        print(f"\n📊 Performance Insights ({time.strftime('%H:%M:%S')}):")
                        for insight in insights:
                            print(f"   {insight}")
                    
                    time.sleep(analysis_interval_seconds)
                except Exception as e:
                    print(f"Analysis loop error: {e}")
                    time.sleep(1.0)
        
        self.analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self.analysis_thread.start()
        print("✅ Continuous performance analysis started")

    def stop_continuous_analysis(self):
        """Stop continuous analysis"""
        self.is_analyzing = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1.0)
        print("🛑 Continuous performance analysis stopped")

    def export_performance_report(self, window_minutes: int = 60, filename: str = None) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        current_time = time.time()
        
        # Generate comprehensive report
        report = {
            'report_timestamp': current_time,
            'window_minutes': window_minutes,
            'summary_metrics': self.get_real_time_metrics(window_minutes),
            'venue_performance': dict(self.get_venue_performance_ranking(window_minutes)),
            'strategy_performance': self.get_strategy_performance_analysis(window_minutes),
            'performance_insights': self.get_performance_insights(window_minutes),
            'benchmark_comparison': self._compare_to_benchmarks(),
            'recommendations': self._generate_optimization_recommendations()
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Performance report exported to {filename}")
        
        return report

    def _compare_to_benchmarks(self) -> Dict[str, Any]:
        """Compare current performance to benchmarks"""
        metrics = self.get_real_time_metrics(15)  # 15-minute window
        
        comparison = {}
        
        # Execution latency comparison
        if 'p50_execution_latency_us' in metrics:
            p50_latency = metrics['p50_execution_latency_us']
            benchmark_p50 = self.benchmark_metrics['execution_latency_p50']
            comparison['latency_p50_vs_benchmark'] = {
                'current': p50_latency,
                'benchmark': benchmark_p50,
                'performance': 'above' if p50_latency > benchmark_p50 else 'meeting',
                'difference_pct': (p50_latency - benchmark_p50) / benchmark_p50 * 100
            }
        
        # Slippage comparison
        if 'p50_slippage_bps' in metrics:
            p50_slippage = metrics['p50_slippage_bps']
            benchmark_p50 = self.benchmark_metrics['slippage_p50']
            comparison['slippage_p50_vs_benchmark'] = {
                'current': p50_slippage,
                'benchmark': benchmark_p50,
                'performance': 'above' if p50_slippage > benchmark_p50 else 'meeting',
                'difference_pct': (p50_slippage - benchmark_p50) / benchmark_p50 * 100
            }
        
        return comparison

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Analyze venue performance
        venue_rankings = self.get_venue_performance_ranking(30)
        if len(venue_rankings) > 2:
            best_venues = venue_rankings[:2]
            worst_venue = venue_rankings[-1]
            
            recommendations.append(f"Focus routing on top performers: {[v[0] for v in best_venues]}")
            if worst_venue[1]['avg_latency_us'] > 2000:
                recommendations.append(f"Consider reducing allocation to {worst_venue[0]} due to high latency")
        
        # Analyze time-based patterns
        current_metrics = self.get_real_time_metrics(5)
        longer_metrics = self.get_real_time_metrics(30)
        
        if current_metrics and longer_metrics:
            latency_trend = current_metrics.get('avg_execution_latency_us', 0) - longer_metrics.get('avg_execution_latency_us', 0)
            if latency_trend > 500:
                recommendations.append("Execution latency trending up - check network conditions or reduce order frequency")
        
        # Strategy recommendations
        strategy_analysis = self.get_strategy_performance_analysis(30)
        profitable_strategies = [s for s, stats in strategy_analysis.items() if stats['total_pnl'] > 0]
        
        if profitable_strategies:
            recommendations.append(f"Most profitable strategies: {profitable_strategies}")
        
        return recommendations


# Integration with trading system
def integrate_performance_analytics(trading_system) -> RealTimePerformanceAnalyzer:
    """Integrate performance analytics with trading system"""
    
    analyzer = RealTimePerformanceAnalyzer()
    
    # Set up analytics callback for interesting insights
    def print_analytics_callback(execution: TradeExecution, metrics: Dict[str, float]):
        # Print significant events
        if execution.execution_latency_us > 3000:
            print(f"⚠️ High latency execution: {execution.execution_latency_us:.0f}μs on {execution.venue}")
        
        if abs(execution.slippage_bps) > 10:
            print(f"💸 High slippage: {execution.slippage_bps:.1f}bps on {execution.symbol}")
    
    analyzer.add_analytics_callback(print_analytics_callback)
    
    # Monkey patch the trading system to record executions
    if hasattr(trading_system, 'execute_order'):
        original_execute = trading_system.execute_order
        
        def enhanced_execute_order(order, market_state, *args, **kwargs):
            start_time = time.time()
            result = original_execute(order, market_state, *args, **kwargs)
            end_time = time.time()
            
            if result:  # If execution was successful
                execution = TradeExecution(
                    trade_id=getattr(result, 'fill_id', f"trade_{int(time.time()*1000)}"),
                    timestamp=end_time,
                    symbol=order.symbol,
                    venue=order.venue,
                    side=order.side.value if hasattr(order.side, 'value') else str(order.side),
                    quantity=order.quantity,
                    requested_price=getattr(order, 'price', market_state.get('mid_price', 0)),
                    executed_price=result.price,
                    arrival_price=market_state.get('mid_price', result.price),
                    execution_latency_us=(end_time - start_time) * 1e6,
                    slippage_bps=getattr(result, 'slippage_bps', 0),
                    market_impact_bps=getattr(result, 'market_impact_bps', 0),
                    fees=getattr(result, 'fees', 0),
                    strategy=getattr(order, 'strategy', 'unknown'),
                    predicted_latency_us=getattr(order, 'predicted_latency_us', None),
                    ml_confidence=getattr(order, 'ml_confidence', None)
                )
                
                analyzer.record_trade_execution(execution)
            
            return result
        
        trading_system.execute_order = enhanced_execute_order
    
    # Start continuous analysis
    analyzer.start_continuous_analysis()
    
    print("✅ Performance analytics integrated with trading system")
    return analyzer


if __name__ == "__main__":
    # Test the performance analyzer
    print("🧪 Testing Real-Time Performance Analyzer...")
    
    analyzer = RealTimePerformanceAnalyzer()
    
    # Simulate some trade executions
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    venues = ['NYSE', 'NASDAQ', 'CBOE', 'IEX', 'ARCA']
    strategies = ['market_making', 'arbitrage', 'momentum']
    
    for i in range(50):
        symbol = np.random.choice(symbols)
        venue = np.random.choice(venues)
        strategy = np.random.choice(strategies)
        
        base_price = 150.0 + np.random.normal(0, 10)
        side = np.random.choice(['buy', 'sell'])
        quantity = int(np.random.exponential(1000))
        
        # Simulate execution with some realistic parameters
        execution_latency = max(200, np.random.lognormal(np.log(800), 0.5))
        slippage = np.random.normal(0, 2)  # Can be positive or negative
        
        execution = TradeExecution(
            trade_id=f"test_trade_{i}",
            timestamp=time.time() - np.random.uniform(0, 300),  # Random time in last 5 minutes
            symbol=symbol,
            venue=venue,
            side=side,
            quantity=quantity,
            requested_price=base_price,
            executed_price=base_price + (slippage * base_price / 10000),
            arrival_price=base_price,
            execution_latency_us=execution_latency,
            slippage_bps=slippage,
            market_impact_bps=abs(slippage) * 0.5,
            fees=quantity * base_price * 0.0001,  # 1bp fee
            strategy=strategy
        )
        
        analyzer.record_trade_execution(execution)
    
    # Get performance insights
    metrics = analyzer.get_real_time_metrics(10)
    insights = analyzer.get_performance_insights(10)
    venue_rankings = analyzer.get_venue_performance_ranking(10)
    
    print(f"\n📊 Performance Summary:")
    print(f"   Avg Execution Latency: {metrics.get('avg_execution_latency_us', 0):.0f}μs")
    print(f"   Avg Slippage: {metrics.get('avg_slippage_bps', 0):.1f}bps")
    print(f"   Total P&L: ${metrics.get('total_pnl', 0):.2f}")
    print(f"   Trade Count: {metrics.get('total_trades', 0)}")
    
    print(f"\n🏆 Top Venue: {venue_rankings[0][0]} (Score: {venue_rankings[0][1]['composite_score']:.1f})")
    
    print(f"\n💡 Key Insights:")
    for insight in insights:
        print(f"   {insight}")
    
    # Export report
    report = analyzer.export_performance_report(30, 'test_performance_report.json')
    
    print("✅ Performance analyzer test complete!")
    print(f"📄 Report exported with {len(report['performance_insights'])} insights")