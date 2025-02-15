# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from telegram import Bot
import json
import os
from ta import momentum, trend, volume, volatility

# Assume config.py exists and contains RISK_PARAMS and SYSTEM_PARAMS
from config import RISK_PARAMS, SYSTEM_PARAMS

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingUtils:
    def __init__(self, telegram_token, chat_id):
        self.telegram_notifier = TelegramNotifier(token=telegram_token, chat_id=chat_id)

    def send_telegram_message(self, message):
        """Send message via Telegram synchronously"""
        self.telegram_notifier.send_message(message)

    @staticmethod
    def calculate_indicators(df):
        """Calculate technical indicators with proper error handling"""
        try:
            # Convert numpy array to pandas DataFrame with correct column names
            if isinstance(df, np.ndarray):
                df = pd.DataFrame(df, columns=[
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'spread', 'real_volume'
                ])
                # Use tick_volume as volume for calculations
                df['volume'] = df['tick_volume']

            if len(df) < 14:  # Minimum data points needed for RSI
                return df

            # Calculate indicators
            df['rsi'] = momentum.rsi(df['close'], window=14)
            df['macd'] = trend.macd_diff(
                df['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            df['atr'] = volatility.average_true_range(
                df['high'],
                df['low'],
                df['close'],
                window=14
            )

            # Volume indicators
            df['obv'] = volume.on_balance_volume(df['close'], df['volume'])
            df['mfi'] = volume.money_flow_index(
                df['high'],
                df['low'],
                df['close'],
                df['volume'],
                window=14
            )

            # Fill NaN values properly
            df = df.ffill().bfill()
            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df

class RiskManager:
    def __init__(self, max_risk_per_trade=0.02, max_daily_loss=0.05, use_trailing_stop=True, atr_multiplier=1.5):
        self.params = RISK_PARAMS
        self.params['position_sizing']['risk_per_trade'] = max_risk_per_trade
        self.params['max_drawdown'] = max_daily_loss
        self.use_trailing_stop = use_trailing_stop
        self.params['position_sizing']['atr_multiplier'] = atr_multiplier
        self.position_history = []
        self.var_history = []

    def calculate_position_size(self, account_info, symbol_info, signal_strength):
        """Calculate optimal position size based on multiple factors"""
        try:
            account_balance = float(account_info.balance)

            # Base position size from risk percentage
            risk_amount = account_balance * self.params['position_sizing']['risk_per_trade']

            # ATR-based position sizing
            atr = symbol_info.get('atr', 0)
            if atr > 0:
                position_size = risk_amount / (atr * self.params['position_sizing']['atr_multiplier'])
            else:
                position_size = risk_amount / 100  # Fallback

            # Adjust for signal strength
            position_size *= abs(signal_strength)

            # Apply limits
            position_size = min(
                position_size,
                self.params['position_sizing']['max_position_size']
            )

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position size calculation error: {str(e)}")
            return 0.0

    def calculate_var(self, positions, returns_data):
        """Calculate Value at Risk"""
        try:
            if not positions or returns_data.empty:
                return 0.0

            # Calculate portfolio returns
            portfolio_returns = self.calculate_portfolio_returns(positions, returns_data)

            # Calculate VaR
            var = np.percentile(portfolio_returns, 
                              (1 - self.params['var_confidence']) * 100)

            self.var_history.append({
                'timestamp': datetime.now(),
                'var': var
            })

            return var

        except Exception as e:
            logger.error(f"VaR calculation error: {str(e)}")
            return 0.0

    def calculate_portfolio_returns(self, positions, returns_data):
        """Calculate portfolio returns for risk analysis"""
        try:
            portfolio_returns = []

            for position in positions:
                symbol_returns = returns_data.get(position['symbol'], [])
                weight = position['volume'] / sum(p['volume'] for p in positions)
                portfolio_returns.append(symbol_returns * weight)

            return np.sum(portfolio_returns, axis=0)

        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {str(e)}")
            return np.array([])

    def check_risk_limits(self, positions, account_info):
        """Check if current positions exceed risk limits"""
        try:
            # Check maximum positions
            if len(positions) >= self.params['max_positions']:
                return False, "Maximum positions reached"

            # Check VaR exposure
            current_var = self.calculate_var(positions, self._get_returns_data())
            if abs(current_var) > self.params['max_var_exposure']:
                return False, "VaR exposure exceeded"

            # Check drawdown
            equity = float(account_info.equity)
            balance = float(account_info.balance)
            drawdown = (balance - equity) / balance

            if drawdown > self.params['max_drawdown']:
                return False, "Maximum drawdown exceeded"

            return True, "Risk checks passed"

        except Exception as e:
            logger.error(f"Risk limit check error: {str(e)}")
            return False, str(e)

    def _get_returns_data(self):
        """Get historical returns data for risk calculations"""
        try:
            # Implementation depends on data storage method
            return pd.DataFrame()  # Placeholder
        except Exception as e:
            logger.error(f"Returns data retrieval error: {str(e)}")
            return pd.DataFrame()

class SystemMonitor:
    def __init__(self):
        self.params = SYSTEM_PARAMS
        self.error_count = 0
        self.last_backup = datetime.now()
        self.system_status = {
            'mt5_connection': False,
            'telegram_connection': False,
            'error_rate': 0.0,
            'last_signal_time': None
        }

    def check_system_health(self):
        """Check overall system health"""
        try:
            health_status = {
                'status': 'healthy',
                'warnings': [],
                'errors': []
            }

            # Check MT5 connection
            if not self.system_status['mt5_connection']:
                health_status['errors'].append('MT5 connection lost')
                health_status['status'] = 'error'

            # Check error rate
            if self.system_status['error_rate'] > 0.1:  # 10% error threshold
                health_status['warnings'].append('High error rate detected')
                health_status['status'] = 'warning'

            # Check signal generation
            if self.system_status['last_signal_time']:
                time_since_last_signal = (datetime.now() - 
                                        self.system_status['last_signal_time'])
                if time_since_last_signal.seconds > 3600:  # 1 hour
                    health_status['warnings'].append('No recent signals generated')

            return health_status

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {'status': 'error', 'errors': [str(e)]}

    def handle_error(self, error):
        """Handle system errors"""
        try:
            self.error_count += 1

            # Log error
            logger.error(f"System error: {str(error)}")

            # Update error rate
            total_operations = max(1, self.error_count + self._get_successful_operations())
            self.system_status['error_rate'] = self.error_count / total_operations

            # Check if recovery is needed
            if self.error_count >= self.params['max_retries']:
                self._initiate_recovery()

            return self.error_count >= self.params['max_retries']

        except Exception as e:
            logger.error(f"Error handling failed: {str(e)}")
            return True

    def _initiate_recovery(self):
        """Initiate system recovery"""
        try:
            logger.info("Initiating system recovery...")

            # Backup current state
            self._backup_system_state()

            # Reset error count
            self.error_count = 0

            # Additional recovery steps can be added here

        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")

    def _backup_system_state(self):
        """Backup system state"""
        try:
            if not os.path.exists('backups'):
                os.makedirs('backups')

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Backup trading state
            state = {
                'system_status': self.system_status,
                'error_count': self.error_count,
                'timestamp': timestamp
            }

            with open(f'backups/system_state_{timestamp}.json', 'w') as f:
                json.dump(state, f)

            self.last_backup = datetime.now()

        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")

    def _get_successful_operations(self):
        """Get count of successful operations"""
        try:
            # Implementation depends on tracking method
            return 100  # Placeholder
        except Exception as e:
            logger.error(f"Operation count error: {str(e)}")
            return 0

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        self.bot = None
        self.chat_id = chat_id
        self.setup_bot(token)

    def setup_bot(self, token):
        """Initialize Telegram bot"""
        try:
            if token:
                self.bot = Bot(token=token)
                logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Telegram bot initialization error: {str(e)}")

    def send_message(self, message, parse_mode='Markdown'):
        """Send formatted message"""
        try:
            if self.bot and self.chat_id:
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
        except Exception as e:
            logger.error(f"Message sending error: {str(e)}")

    def send_trade_notification(self, trade_info):
        """Send trade notification"""
        try:
            message = (
                f"🔔 *Trading Signal*\n"
                f"Symbol: `{trade_info['symbol']}`\n"
                f"Action: {'🟢 BUY' if trade_info['action'] == 'buy' else '🔴 SELL'}\n"
                f"Entry: `{trade_info['entry']:.5f}`\n"
                f"Stop Loss: `{trade_info['sl']:.5f}`\n"
                f"Take Profit: `{trade_info['tp']:.5f}`\n"
                f"Volume: `{trade_info['volume']:.2f}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            self.send_message(message)

        except Exception as e:
            logger.error(f"Trade notification error: {str(e)}")

    def send_error_notification(self, error):
        """Send error notification"""
        try:
            message = (
                f"⚠️ *System Error*\n"
                f"Error: `{str(error)}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            self.send_message(message)

        except Exception as e:
            logger.error(f"Error notification failed: {str(e)}")

    def send_performance_update(self, metrics):
        """Send performance metrics update"""
        try:
            message = (
                f"📊 *Performance Update*\n"
                f"Win Rate: `{metrics['win_rate']:.2%}`\n"
                f"Profit Factor: `{metrics.get('profit_factor', 0):.2f}`\n"
                f"Sharpe Ratio: `{metrics.get('sharpe_ratio', 0):.2f}`\n"
                f"Max Drawdown: `{metrics.get('max_drawdown', 0):.2%}`\n"
                f"Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            )
            self.send_message(message)

        except Exception as e:
            logger.error(f"Performance update notification failed: {str(e)}")