try:
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise ImportError("MetaTrader5 initialize failed")
    USING_MOCK = False
    print("✅ MetaTrader5 modülü başarıyla yüklendi ve bağlantı kuruldu")
except ImportError as e:
    print("❌ MetaTrader5 terminal kurulu değil veya bağlantı kurulamadı!")
    print("MetaTrader5 terminal programını yükleyin ve tekrar deneyin.")
    print(f"Hata: {str(e)}")
    raise SystemExit("MetaTrader5 bağlantısı gerekli")

import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading
from config import *
from models import AdvancedTradingModel
from utils import TradingUtils, RiskManager, SystemMonitor
from dashboard import TradingDashboard
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexTrader:
    def __init__(self):
        # Initialize components first
        self.system_monitor = SystemMonitor()
        self.risk_manager = RiskManager()
        self.risk_manager.params['position_sizing']['risk_per_trade'] = 0.01  # %1 risk
        self.risk_manager.params['max_drawdown'] = 0.03  # %3 max loss
        self.model = AdvancedTradingModel()
        # Removed USING_MOCK - connection handled differently now

        # Set up Telegram notifications with environment variables
        self.utils = TradingUtils(
            os.getenv('TELEGRAM_TOKEN'),
            os.getenv('TELEGRAM_CHAT_ID')
        )

        # Initialize dashboard
        self.dashboard = TradingDashboard()

        # Initialize state variables
        self.running = True
        self.performance_metrics = {
            'trades': [],
            'equity': [],
            'drawdown': []
        }

        # Connect to MT5
        self.connect_mt5()

    def connect_mt5(self):
        """Initialize MT5 connection with real broker and enhanced error handling"""
        retries = 3
        for attempt in range(retries):
            try:
                if mt5.initialize(
                    server=MT5_CONFIG['server'],
                    login=int(MT5_CONFIG['login']),
                    password=str(MT5_CONFIG['password'])
                ):
                    account_info = mt5.account_info()
                    if account_info is not None:
                        mode = "Test Mode (Mock MT5)" if False else "Live Trading" #Always Live now
                        logger.info(f"✅ MT5 Bağlantısı başarılı - Mod: {mode}")
                        logger.info(f"Hesap: {account_info.login}")
                        logger.info(f"Bakiye: {account_info.balance} | Equity: {account_info.equity}")
                        self.system_monitor.system_status['mt5_connection'] = True
                        return True

                error = mt5.last_error()
                logger.error(f"MT5 Bağlantı hatası (Deneme {attempt+1}/{retries}): {error}")
                time.sleep(5)  # Retry delay

            except Exception as e:
                logger.error(f"MT5 Bağlantı exception (Deneme {attempt+1}/{retries}): {str(e)}")
                time.sleep(5)

        raise ConnectionError("MT5 bağlantısı başarısız oldu")

    def get_market_data(self, symbol, timeframe, n_bars=1000):
        """Enhanced market data collection with multiple timeframes"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
            if rates is None or len(rates) == 0:
                raise ValueError(f"Failed to get data for {symbol}")

            df = self.utils.calculate_indicators(rates)

            # Add order book analysis if available
            order_book = mt5.market_book_get(symbol)
            if order_book:
                df['bid_volume'] = sum(b['volume'] for b in order_book.bid)
                df['ask_volume'] = sum(a['volume'] for a in order_book.ask)
                df['book_imbalance'] = df['bid_volume'] / (df['ask_volume'] + 1e-6)

            return df

        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None

    def execute_trade(self, symbol, signal, volume):
        """Execute trade with enhanced risk management"""
        try:
            if signal not in ['buy', 'sell'] or volume <= 0:
                logger.warning(f"Invalid trade parameters - Signal: {signal}, Volume: {volume}")
                return False

            # Additional pre-trade checks
            if not self._validate_trading_conditions(symbol):
                return False

            # Get current price info with retry mechanism
            tick = self._get_tick_with_retry(symbol)
            if tick is None:
                return False

            price = float(tick.ask if signal == 'buy' else tick.bid)

            # Enhanced ATR-based SL/TP calculation
            sl, tp = self._calculate_sl_tp(symbol, signal, price)
            if None in (sl, tp):
                return False

            # Final risk check before execution
            if not self._final_risk_check(symbol, volume, price, sl):
                return False

            # Execute trade with retry mechanism
            return self._execute_order(symbol, signal, volume, price, sl, tp)

        except Exception as e:
            logger.error(f"❌ Trade execution error: {str(e)}")
            return False

    def _validate_trading_conditions(self, symbol):
        """Validate pre-trade conditions"""
        try:
            # Check spread
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info.spread > TRADING_PARAMS['max_spread']:
                logger.warning(f"Spread too high for {symbol}: {symbol_info.spread}")
                return False

            # Check market volatility
            market_data = self.get_market_data(symbol, mt5.TIMEFRAME_M5)
            if market_data is None:
                return False

            volatility = market_data['atr'].iloc[-1] / market_data['close'].iloc[-1]
            if volatility > MARKET_PARAMS['volatility_threshold']:
                logger.warning(f"Volatility too high for {symbol}: {volatility:.4f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Trading conditions validation error: {str(e)}")
            return False

    def _get_tick_with_retry(self, symbol, max_retries=3):
        """Get tick data with retry mechanism"""
        for attempt in range(max_retries):
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    return tick
                time.sleep(1)
            except Exception as e:
                logger.error(f"Tick data error (attempt {attempt+1}): {str(e)}")
        return None

    def _calculate_sl_tp(self, symbol, signal, price):
    """
    Calculate dynamic SL/TP levels based on ATR.
    """
    try:
        # ATR hesaplaması için piyasa verisini al
        data = self.get_market_data(symbol, mt5.TIMEFRAME_H1)
        if data is None or data.empty:
            return None, None

        # ATR kullanarak SL/TP hesapla
        atr = float(data['atr'].iloc[-1])
        sl_distance = atr * RISK_PARAMS['position_sizing']['atr_multiplier']
        tp_distance = sl_distance * RISK_PARAMS['position_sizing']['min_risk_reward']

        # Sinyale göre seviyeleri hesapla
        if signal == 'buy':
            sl = price - sl_distance
            tp = price + tp_distance
        else:  # sell
            sl = price + sl_distance
            tp = price - tp_distance

        # Seviyeleri doğrula
        if not self._validate_sl_tp_levels(symbol, signal, price, sl, tp):
            return None, None

        return round(sl, 5), round(tp, 5)

    except Exception as e:
        logger.error(f"SL/TP calculation error: {str(e)}")
        return None, None


def _validate_sl_tp_levels(self, symbol, signal, price, sl, tp):
    """
    Validate SL/TP levels.
    """
    try:
        symbol_info = mt5.symbol_info(symbol)

        # Minimum mesafe kontrolü (örneğin: 20 pip)
        min_distance = symbol_info.point * 20

        if signal == 'buy':
            if (price - sl) < min_distance or (tp - price) < min_distance:
                logger.warning(f"Invalid SL/TP distances for {symbol}")
                return False
        else:  # sell
            if (sl - price) < min_distance or (price - tp) < min_distance:
                logger.warning(f"Invalid SL/TP distances for {symbol}")
                return False

        return True

    except Exception as e:
        logger.error(f"SL/TP validation error: {str(e)}")
        return False



def _validate_sl_tp_levels(self, symbol, signal, price, sl, tp):
    """
    Validate SL/TP levels
    """
    try:
        symbol_info = mt5.symbol_info(symbol)

        # Check minimum distance (örnek: 20 pip)
        min_distance = symbol_info.point * 20

        if signal == 'buy':
            if (price - sl) < min_distance or (tp - price) < min_distance:
                logger.warning(f"Invalid SL/TP distances for {symbol}")
                return False
        else:  # sell
            if (sl - price) < min_distance or (price - tp) < min_distance:
                logger.warning(f"Invalid SL/TP distances for {symbol}")
                return False

        return True

    except Exception as e:
        logger.error(f"SL/TP validation error: {str(e)}")
        return False


    def _final_risk_check(self, symbol, volume, price, sl):
        """Final risk check before trade execution"""
        try:
            # Calculate potential loss
            point_value = mt5.symbol_info(symbol).point
            sl_points = abs(price - sl) / point_value
            potential_loss = sl_points * volume * point_value

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return False

            # Calculate risk percentage
            risk_percent = potential_loss / account_info.balance

            # Check against risk limits
            if risk_percent > TRADING_PARAMS['risk_per_trade']:
                logger.warning(f"Risk too high: {risk_percent:.2%}")
                return False

            # Check daily loss limit
            if self._check_daily_loss_limit(account_info):
                return False

            return True

        except Exception as e:
            logger.error(f"Risk check error: {str(e)}")
            return False

    def _check_daily_loss_limit(self, account_info):
        """Check if daily loss limit is reached"""
        try:
            today_trades = [t for t in self.performance_metrics['trades']
                              if t['timestamp'].date() == datetime.now().date()]

            if not today_trades:
                return False

            daily_loss = sum(t.get('profit', 0) for t in today_trades)
            daily_loss_percent = abs(daily_loss) / account_info.balance

            if daily_loss_percent >= TRADING_PARAMS['max_daily_loss']:
                logger.warning(f"Daily loss limit reached: {daily_loss_percent:.2%}")
                return True

            return False

        except Exception as e:
            logger.error(f"Daily loss check error: {str(e)}")
            return True  # Fail-safe return

    def _execute_order(self, symbol, signal, volume, price, sl, tp):
        """Execute order with proper error handling"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": float(sl),
                "tp": float(tp),
                "deviation": 10,
                "magic": 234000,
                "comment": f"ai-signal-{datetime.now().strftime('%Y%m%d%H%M')}",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            # Execute trade
            result = mt5.order_send(request)

            if result and hasattr(result, 'retcode') and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Log successful trade
                trade_info = {
                    'symbol': symbol,
                    'action': signal,
                    'entry': price,
                    'sl': sl,
                    'tp': tp,
                    'volume': volume
                }
                self.utils.telegram_notifier.send_trade_notification(trade_info)
                logger.info(f"✅ Trade executed successfully: {symbol} {signal.upper()}")
                self.log_trade(symbol, signal, volume, price, sl, tp)
                return True
            else:
                logger.error(f"❌ Trade execution failed: {mt5.last_error()}")
                return False

        except Exception as e:
            logger.error(f"Order execution error: {str(e)}")
            return False

    def log_trade(self, symbol, signal, volume, price, sl, tp):
        """Log trade for performance tracking"""
        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': signal,
            'volume': volume,
            'price': price,
            'sl': sl,
            'tp': tp
        }
        self.performance_metrics['trades'].append(trade)
        self.model.performance_tracker.add_trade(trade)

    def run(self):
        """Main trading loop"""
        try:
            # Start dashboard in separate thread
            dashboard_thread = threading.Thread(
                target=self.dashboard.run,
                kwargs={'host': DASHBOARD_CONFIG['host'], 'port': DASHBOARD_CONFIG['port']}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()

            logger.info("Trading system started successfully")

            # Main trading loop
            while self.running:
                try:
                    self._trading_cycle()
                    time.sleep(60)  # Check every minute

                except Exception as e:
                    logger.error(f"Trading cycle error: {str(e)}")
                    if self.system_monitor.handle_error(e):
                        time.sleep(300)  # Wait 5 minutes on critical error

        except Exception as e:
            logger.error(f"System startup error: {str(e)}")
            self.shutdown()

    def update_account_info(self):
        """Update account information"""
        try:
            account_info = mt5.account_info()
            if account_info is not None:
                logger.info(f"Account Balance Updated - Current Balance: {account_info.balance}")
                return account_info.balance
            return None
        except Exception as e:
            logger.error(f"Error updating account info: {str(e)}")
            return None

    def _trading_cycle(self):
        """Single trading cycle implementation"""
        # Update account information
        current_balance = self.update_account_info()
        if current_balance is None:
            logger.warning("Failed to update account balance")

        # System health check
        health_status = self.system_monitor.check_system_health()
        if health_status['status'] != 'healthy':
            logger.warning(f"System health issues: {health_status}")
            return

        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            raise ValueError("Failed to get account info")

        # Update available symbols if needed
        if not TRADING_PARAMS['symbols']:
            self._update_available_symbols()

        # Process each symbol
        for symbol in TRADING_PARAMS['symbols']:
            try:
                self._process_symbol(symbol, account_info)
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")
                continue

    def _update_available_symbols(self):
        """Update list of available trading symbols"""
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols:
                TRADING_PARAMS['symbols'] = [
                    s.name for s in all_symbols
                    if s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
                ]
                logger.info(f"Updated available symbols: {len(TRADING_PARAMS['symbols'])} symbols found")
            else:
                logger.error("Failed to get symbols from MT5")
        except Exception as e:
            logger.error(f"Error updating symbols: {str(e)}")

    def _process_symbol(self, symbol, account_info):
        """Process a single symbol for trading opportunities"""
        # Skip if spread is too high  (This check is now within _validate_trading_conditions)

        # Get multi-timeframe data
        market_data = self._get_market_data(symbol)
        if not market_data:
            return

        # Generate and process trading signals
        analysis = self.model.analyze_market(market_data)
        if abs(analysis) >= TRADING_PARAMS['confidence_threshold']:
            self._execute_signal(symbol, analysis, account_info, market_data)

    def _get_market_data(self, symbol):
        """Get multi-timeframe market data for analysis"""
        market_data = {}
        for tf in TRADING_PARAMS['timeframes']:
            data = self.get_market_data(symbol, getattr(mt5, f'TIMEFRAME_{tf}'))
            if data is not None:
                market_data[tf] = data
        return market_data

    def _execute_signal(self, symbol, analysis, account_info, market_data):
        """Execute trading signal if conditions are met"""
        signal = 'buy' if analysis > 0 else 'sell'

        # Calculate position size
        volume = self.risk_manager.calculate_position_size(
            account_info,
            {'atr': market_data['H1']['atr'].iloc[-1]},
            abs(analysis)
        )

        if volume > 0:
            success = self.execute_trade(symbol, signal, volume)
            if success:
                logger.info(f"Successfully executed {signal} trade for {symbol}")

    def shutdown(self):
        """Clean shutdown of the trading system"""
        logger.info("Shutting down trading system...")
        self.running = False
        mt5.shutdown()
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    trader = ForexTrader()
    trader.run()