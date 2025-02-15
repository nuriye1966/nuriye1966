# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import ta
import logging
from config import TECHNICAL_PARAMS, RISK_PARAMS, SYSTEM_PARAMS, MARKET_PARAMS

# Loglama konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedTradingModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.technical_indicators = TechnicalIndicators()
        self.market_analyzer = MarketAnalyzer()
        self.signal_generator = SignalGenerator()
        self.performance_tracker = PerformanceTracker()

    def analyze_market(self, data_dict):
        """Comprehensive market analysis"""
        try:
            if not data_dict:
                logger.warning("Empty data dictionary")
                return 0

            # Process all timeframes
            analysis_results = {}
            for timeframe, data in data_dict.items():
                if data is None or not isinstance(data, (np.ndarray, pd.DataFrame)) or len(data) == 0:
                    logger.warning(f"Invalid data for timeframe {timeframe}")
                    continue

                # Calculate technical indicators
                indicators = self.technical_indicators.calculate_all(data)
                if indicators is None:
                    continue

                # Market microstructure analysis
                market_analysis = self.market_analyzer.analyze(data)
                if market_analysis is None:
                    continue

                # Combine analyses
                analysis_results[timeframe] = {
                    'technical': indicators,
                    'market': market_analysis
                }

            if not analysis_results:
                logger.warning("No valid analysis results")
                return 0

            # Generate final signal
            signal = self.signal_generator.generate_signals(analysis_results)
            logger.info(f"Generated signal strength: {signal}")
            return signal

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return 0


class TechnicalIndicators:
    def __init__(self):
        self.params = TECHNICAL_PARAMS

    def calculate_all(self, df):
        """Calculate all technical indicators"""
        try:
            # Convert numpy array to pandas DataFrame with proper column names
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df, columns=[
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'spread', 'real_volume'
                ])
                # Use tick_volume as volume for calculations
                df['volume'] = df['tick_volume']

            if len(df) < 30:  # Need enough data points for all indicators
                logger.warning("Insufficient data points for indicators")
                return None

            results = {}

            # Trend Indicators
            results['rsi'] = ta.momentum.RSIIndicator(
                close=df['close'], 
                window=self.params['rsi']['period']
            ).rsi()

            macd = ta.trend.MACD(
                close=df['close'],
                window_slow=self.params['macd']['slow_period'],
                window_fast=self.params['macd']['fast_period'],
                window_sign=self.params['macd']['signal_period']
            )
            results['macd'] = {'macd': macd.macd_diff()}

            # Volatility Indicators
            results['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], 
                low=df['low'], 
                close=df['close'],
                window=self.params['atr']['period']
            ).average_true_range()

            results['bollinger'] = self.calculate_bollinger_bands(df)

            # Volume Analysis
            results['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'], 
                volume=df['volume']
            ).on_balance_volume()

            results['mfi'] = ta.volume.MFIIndicator(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume'],
                window=14
            ).money_flow_index()

            # Momentum
            results['momentum'] = self.calculate_momentum(df)

            # Fill NaN values using forward fill then backward fill
            for key in results:
                if isinstance(results[key], pd.Series):
                    results[key] = results[key].ffill().bfill()
                elif isinstance(results[key], dict):
                    for subkey in results[key]:
                        if isinstance(results[key][subkey], pd.Series):
                            results[key][subkey] = results[key][subkey].ffill().bfill()

            # Verify no NaN values remain
            for key in results:
                if isinstance(results[key], pd.Series):
                    if results[key].isna().any():
                        logger.warning(f"NaN values remain in {key}")
                        return None
                elif isinstance(results[key], dict):
                    for subkey in results[key]:
                        if isinstance(results[key][subkey], pd.Series) and results[key][subkey].isna().any():
                            logger.warning(f"NaN values remain in {key}.{subkey}")
                            return None

            return results

        except Exception as e:
            logger.error(f"Technical indicator calculation error: {str(e)}")
            return None

    def calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands"""
        try:
            indicator_bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=self.params['bollinger']['period'],
                window_dev=self.params['bollinger']['std_dev']
            )

            return {
                'middle': indicator_bb.bollinger_mavg(),
                'upper': indicator_bb.bollinger_hband(),
                'lower': indicator_bb.bollinger_lband()
            }

        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {str(e)}")
            return None

    def calculate_momentum(self, df):
        """Calculate various momentum indicators"""
        try:
            return {
                'roc': ta.momentum.ROCIndicator(
                    close=df['close'], 
                    window=self.params['momentum']['roc_period']
                ).roc(),
                'cci': ta.trend.CCIIndicator(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=self.params['momentum']['cci_period']
                ).cci(),
                'willr': ta.momentum.WilliamsRIndicator(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    lbp=self.params['momentum']['willr_period']
                ).williams_r()
            }
        except Exception as e:
            logger.error(f"Momentum calculation error: {str(e)}")
            return None


class MarketAnalyzer:
    def __init__(self):
        self.params = MARKET_PARAMS

    def analyze(self, df):
        """Analyze market microstructure"""
        try:
            # Convert numpy array to pandas DataFrame if needed
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df, columns=[
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'spread', 'real_volume'
                ])

            results = {}

            # Volume Analysis
            results['volume_ma'] = df['tick_volume'].rolling(window=20, min_periods=1).mean()
            results['volume_std'] = df['tick_volume'].rolling(window=20, min_periods=1).std()

            # Volatility Analysis
            returns = df['close'].pct_change()
            results['volatility'] = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)

            # Price Analysis
            results['price_momentum'] = returns.rolling(window=10, min_periods=1).mean()
            results['price_acceleration'] = results['price_momentum'].diff()

            # Spread Analysis
            results['spread'] = (df['high'] - df['low']) / df['low']

            # Fill NaN values
            for key in results:
                results[key] = results[key].ffill().bfill()
                if results[key].isna().any():
                    logger.warning(f"NaN values remain in market analysis {key}")
                    return None

            return results

        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return None


class SignalGenerator:
    def __init__(self):
        self.last_signal = None
        self.signal_counter = 0

    def generate_signals(self, analysis_results):
        """Generate trading signals based on multi-timeframe analysis"""
        try:
            if not analysis_results:
                return 0

            signals = {}
            weights = {'M1': 0.2, 'M5': 0.3, 'H1': 0.5}

            for timeframe, analysis in analysis_results.items():
                technical = analysis.get('technical')
                market = analysis.get('market')

                if technical is None or market is None:
                    continue

                # Calculate individual signals
                trend_signal = self.analyze_trend(technical)
                momentum_signal = self.analyze_momentum(technical)
                volatility_signal = self.analyze_volatility(technical, market)
                volume_signal = self.analyze_volume(technical, market)

                # Skip if any signal is None
                if any(s is None for s in [trend_signal, momentum_signal, volatility_signal, volume_signal]):
                    continue

                # Weighted signal combination
                signals[timeframe] = sum([
                    trend_signal * 0.4,
                    momentum_signal * 0.3,
                    volatility_signal * 0.2,
                    volume_signal * 0.1
                ]) * weights.get(timeframe, 0.3)

            # Return 0 if no valid signals
            if not signals:
                return 0

            # Final signal decision
            final_signal = sum(signals.values())

            # Signal smoothing and filtering
            return self.filter_signal(final_signal)

        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            return 0

    def analyze_trend(self, technical):
        """Analyze trend indicators"""
        try:
            rsi = technical.get('rsi')
            macd = technical.get('macd', {}).get('macd')

            if rsi is None or macd is None:
                return None

            rsi_value = rsi.iloc[-1]
            macd_value = macd.iloc[-1]

            trend_score = 0

            # RSI Analysis
            if rsi_value < 30:
                trend_score += 1
            elif rsi_value > 70:
                trend_score -= 1

            # MACD Analysis
            if macd_value > 0:
                trend_score += 0.5
            else:
                trend_score -= 0.5

            return np.clip(trend_score, -1, 1)

        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return None

    def analyze_momentum(self, technical):
        """Analyze momentum indicators"""
        try:
            momentum = technical.get('momentum', {})
            if not momentum:
                return 0

            values = []
            weights = []
            
            if 'roc' in momentum and len(momentum['roc']) > 0:
                values.append(momentum['roc'].iloc[-1])
                weights.append(0.4)
                
            if 'cci' in momentum and len(momentum['cci']) > 0:
                values.append(momentum['cci'].iloc[-1])
                weights.append(0.3)
                
            if 'willr' in momentum and len(momentum['willr']) > 0:
                values.append(momentum['willr'].iloc[-1])
                weights.append(0.3)

            if not values:
                return 0

            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Calculate weighted score
            score = np.sum(np.array(values) * weights)
            
            return np.clip(score / 100, -1, 1)  # Normalize to [-1, 1]

        except Exception as e:
            logger.error(f"Momentum analysis error: {str(e)}")
            return None

    def analyze_volatility(self, technical, market):
        """Analyze volatility conditions"""
        try:
            if 'atr' not in technical or 'volatility' not in market:
                return 0

            atr = technical['atr'].iloc[-1] if len(technical['atr']) > 0 else 0
            volatility = market['volatility'].iloc[-1] if len(market['volatility']) > 0 else 0

            if pd.isna(atr) or pd.isna(volatility):
                return 0

            # Normalize volatility score
            vol_score = -np.clip(volatility / 0.02, -1, 1)  # 0.02 = 2% volatility threshold

            return vol_score

        except Exception as e:
            logger.error(f"Volatility analysis error: {str(e)}")
            return 0

    def analyze_volume(self, technical, market):
        """Analyze volume indicators"""
        try:
            score = 0
            count = 0
            
            if 'obv' in technical and len(technical['obv']) > 0:
                obv = technical['obv'].iloc[-1]
                if not pd.isna(obv):
                    score += np.clip(obv / 1000000, -1, 1) * 0.4
                    count += 0.4
                    
            if 'mfi' in technical and len(technical['mfi']) > 0:
                mfi = technical['mfi'].iloc[-1]
                if not pd.isna(mfi):
                    score += ((mfi - 50) / 50) * 0.3
                    count += 0.3
                    
            if 'volume_ma' in market and len(market['volume_ma']) > 0:
                vol = market['volume_ma'].iloc[-1]
                if not pd.isna(vol):
                    score += np.clip(vol / 1000, -1, 1) * 0.3
                    count += 0.3

            if count == 0:
                return 0
                
            return np.clip(score / count, -1, 1)

        except Exception as e:
            logger.error(f"Volume analysis error: {str(e)}")
            return None

    def filter_signal(self, signal):
        """Filter and smooth signals to prevent excessive trading"""
        try:
            # Signal threshold
            if abs(signal) < 0.3:  # Minimum signal strength
                return 0

            # Prevent consecutive signals
            if self.last_signal is not None:
                if (signal > 0 and self.last_signal > 0) or \
                   (signal < 0 and self.last_signal < 0):
                    self.signal_counter += 1
                    if self.signal_counter > 3:  # Max consecutive signals
                        return 0
                else:
                    self.signal_counter = 0

            self.last_signal = signal
            return signal

        except Exception as e:
            logger.error(f"Signal filtering error: {str(e)}")
            return 0


class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.metrics = {}

    def add_trade(self, trade):
        """Add trade to performance tracking"""
        self.trades.append(trade)
        self.update_metrics()

    def update_metrics(self):
        """Update performance metrics"""
        try:
            if not self.trades:
                return

            returns = [trade.get('profit', 0) for trade in self.trades]

            self.metrics = {
                'total_trades': len(self.trades),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'avg_return': np.mean(returns),
                'sharpe_ratio': self.calculate_sharpe(returns),
                'max_drawdown': self.calculate_max_drawdown(returns)
            }

        except Exception as e:
            logger.error(f"Metrics update error: {str(e)}")

    def calculate_sharpe(self, returns):
        """Calculate Sharpe Ratio"""
        try:
            if not returns or len(returns) < 2:
                return 0
            return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        except Exception as e:
            logger.error(f"Sharpe calculation error: {str(e)}")
            return 0

    def calculate_max_drawdown(self, returns):
        """Calculate Maximum Drawdown"""
        try:
            if not returns:
                return 0
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        except Exception as e:
            logger.error(f"Drawdown calculation error: {str(e)}")
            return 0