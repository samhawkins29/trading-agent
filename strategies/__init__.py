"""
Trading strategies — redesigned for evidence-based quantitative trading.

Strategy lineup:
  - MeanReversionStrategy: Statistical Arbitrage / Pairs Trading
  - MomentumStrategy: Time Series Momentum (Moskowitz et al.)
  - SentimentStrategy: Factor Momentum (price-based Fama-French proxies)
  - PatternRecognitionStrategy: Volatility-Regime Detection
  - DualMomentumStrategy: Antonacci GEM + Keller VAA breadth momentum
  - CrossAssetSignalStrategy: Cross-asset vol/trend risk overlay
  - AdaptiveAllocationStrategy: Risk parity + adaptive weight optimization
"""

from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
from strategies.sentiment import SentimentStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from strategies.dual_momentum import DualMomentumStrategy
from strategies.cross_asset_signals import CrossAssetSignalStrategy
from strategies.adaptive_allocation import AdaptiveAllocationStrategy

__all__ = [
    "MeanReversionStrategy",
    "MomentumStrategy",
    "SentimentStrategy",
    "PatternRecognitionStrategy",
    "DualMomentumStrategy",
    "CrossAssetSignalStrategy",
    "AdaptiveAllocationStrategy",
]
