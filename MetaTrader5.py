import random
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Only log errors
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def last_error():
    """Return last error message"""
    return "No error"

class MockMT5:
    def __init__(self):
        self.initialized = False
        self.test_data = self._generate_test_data()
        self._last_error = None
        self.account_name = "nuriye demoo"

    def initialize(self, server=None, login=None, password=None):
        """Initialize connection with credentials check"""
        try:
            if not all([server, login, password]):
                self._last_error = "Missing credentials"
                return False

            # Validate credentials against config
            if (server == "MetaQuotes-Demo" and 
                str(login) == "5033641077" and 
                password == "@7DiRcLx"):
                print(f"✅ Successfully connected to {server} - Account: {self.account_name}")
                self.initialized = True
                return True
            else:
                self._last_error = "Invalid credentials"
                return False
        except Exception as e:
            self._last_error = str(e)
            return False

    def shutdown(self):
        self.initialized = False
        return True

    def _generate_test_data(self):
        """Generate sample OHLCV data"""
        try:
            dates = [datetime.now() - timedelta(minutes=i) for i in range(1000)]
            base_price = 1.1000  # Example EUR/USD price
            data = []

            for date in dates:
                price = base_price + random.uniform(-0.002, 0.002)
                data.append((
                    int(date.timestamp()),  # time
                    float(price),  # open
                    float(price + random.uniform(0, 0.001)),  # high
                    float(price - random.uniform(0, 0.001)),  # low
                    float(price + random.uniform(-0.001, 0.001)),  # close
                    int(random.randint(100, 1000)),  # tick_volume
                    int(random.randint(1, 3)),  # spread
                    int(random.randint(10000, 100000))  # real_volume
                ))

            return np.array(data, dtype=[
                ('time', '<i8'),
                ('open', '<f8'),
                ('high', '<f8'),
                ('low', '<f8'),
                ('close', '<f8'),
                ('tick_volume', '<i8'),
                ('spread', '<i8'),
                ('real_volume', '<i8')
            ])
        except Exception as e:
            logger.error(f"Error generating test data: {str(e)}")
            return np.array([], dtype=[
                ('time', '<i8'), ('open', '<f8'), ('high', '<f8'),
                ('low', '<f8'), ('close', '<f8'), ('tick_volume', '<i8'),
                ('spread', '<i8'), ('real_volume', '<i8')
            ])

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Return OHLCV data as numpy array with correct structure"""
        try:
            if not self.initialized:
                logger.error("MT5 not initialized")
                return None

            if count <= 0:
                logger.error("Invalid count parameter")
                return None

            return self.test_data[:count]
        except Exception as e:
            logger.error(f"Error generating mock data: {str(e)}")
            return None

    def account_info(self):
        """Mock account information with correct details"""
        try:
            if not self.initialized:
                return None

            class AccountInfo:
                def __init__(self):
                    self.login = int(5033641077)
                    self.name = "nuriye demoo"
                    self.server = "MetaQuotes-Demo"
                    self.balance = float(100000.00)
                    self.equity = float(100000.00)
                    self.margin = float(0.0)
                    self.profit = float(0.0)
            return AccountInfo()
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def symbol_info(self, symbol):
        """Mock symbol information with error handling"""
        try:
            if not self.initialized:
                return None

            class SymbolInfo:
                def __init__(self):
                    self.spread = int(2)
                    self.point = float(0.00001)
                    self.atr = float(0.001)
                    self.name = symbol
                    self.trade_mode = SYMBOL_TRADE_MODE_FULL
            return SymbolInfo()
        except Exception as e:
            logger.error(f"Error getting symbol info: {str(e)}")
            return None

    def symbol_info_tick(self, symbol):
        """Mock tick information with error handling"""
        try:
            if not self.initialized:
                return None

            class TickInfo:
                def __init__(self):
                    self.bid = float(1.1000 + random.uniform(-0.0001, 0.0001))
                    self.ask = float(self.bid + 0.0002)
                    self.last = float(self.bid)
                    self.volume = int(random.randint(1, 100))
            return TickInfo()
        except Exception as e:
            logger.error(f"Error getting tick info: {str(e)}")
            return None

    def positions_get(self):
        """Mock positions with error handling"""
        try:
            if not self.initialized:
                return None
            return []
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None

    def order_send(self, request):
        """Mock order sending with error handling"""
        try:
            if not self.initialized:
                return None

            class OrderResult:
                def __init__(self):
                    self.retcode = TRADE_RETCODE_DONE
                    self.comment = "Order executed successfully"
                    self.volume = float(request['volume'])
                    self.price = float(request['price'])
                    self.symbol = request['symbol']
            
            return OrderResult()
        except Exception as e:
            logger.error(f"Error sending order: {str(e)}")
            return None

    def market_book_get(self, symbol):
        """Mock market depth information with error handling"""
        try:
            if not self.initialized:
                return None

            class MarketBook:
                def __init__(self):
                    self.bid = [{'volume': int(random.randint(1, 100))} for _ in range(5)]
                    self.ask = [{'volume': int(random.randint(1, 100))} for _ in range(5)]
            
            return MarketBook()
        except Exception as e:
            logger.error(f"Error getting market book: {str(e)}")
            return None

    def symbols_get(self):
        """Mock symbol retrieval with error handling"""
        try:
            if not self.initialized:
                return None

            valid_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
            
            class SymbolInfo:
                def __init__(self, name):
                    self.name = name
                    self.trade_mode = SYMBOL_TRADE_MODE_FULL
            
            return [SymbolInfo(s) for s in valid_symbols]
        except Exception as e:
            logger.error(f"Sembol listesi alım hatası: {str(e)}")
            return None

# Create a global instance
instance = MockMT5()

# Module level functions
def initialize(*args, **kwargs):
    return instance.initialize(*args, **kwargs)

def shutdown():
    return instance.shutdown()

def copy_rates_from_pos(*args, **kwargs):
    return instance.copy_rates_from_pos(*args, **kwargs)

def account_info():
    return instance.account_info()

def symbol_info(*args, **kwargs):
    return instance.symbol_info(*args, **kwargs)

def symbol_info_tick(*args, **kwargs):
    return instance.symbol_info_tick(*args, **kwargs)

def positions_get():
    return instance.positions_get()

def order_send(*args, **kwargs):
    return instance.order_send(*args, **kwargs)

def market_book_get(*args, **kwargs):
    return instance.market_book_get(*args, **kwargs)

def symbols_get(*args, **kwargs):
    return instance.symbols_get()

# Constants
TIMEFRAME_M1 = "M1"
TIMEFRAME_M5 = "M5"
TIMEFRAME_H1 = "H1"
TIMEFRAME_D1 = "D1"
ORDER_TYPE_BUY = "BUY"
ORDER_TYPE_SELL = "SELL"
TRADE_ACTION_DEAL = "DEAL"
ORDER_TIME_GTC = "GTC"
TRADE_RETCODE_DONE = 10009
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
SYMBOL_TRADE_MODE_FULL = 4
