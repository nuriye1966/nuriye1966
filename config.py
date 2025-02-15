# -*- coding: utf-8 -*-
# Advanced Trading System Configuration

# MT5 Configuration
MT5_CONFIG = {
    'server': 'MetaQuotes-Demo',
    'login': 5033641077,
    'password': '@7DiRcLx'
}

# Trading Parameters
TRADING_PARAMS = {
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD'],  # Örnek sembol listesi
    'max_spread': 20,
    'confidence_threshold': 0.3,
    'timeframes': ['M1', 'M5', 'H1'],  # Multiple timeframes for analysis
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_daily_loss': 0.05,  # 5% max daily loss
    'max_positions': 5,  # Maximum concurrent positions
    'confidence_threshold': 0.75,  # Signal confidence threshold
    'min_volume': 0.01,  # Minimum trade volume
    'max_spread': 20,  # Maximum allowed spread in points
    'use_martingale': False,  # Martingale strategy disabled by default
    'martingale_multiplier': 1.5,  # Martingale multiplier if enabled
    'correlation_threshold': 0.7,  # Correlation threshold for pair trading
    'auto_hedge': True,  # Automatic hedging for correlated pairs
}

# Risk Management
RISK_PARAMS = {
    'use_var': True,  # Value at Risk calculation
    'var_confidence': 0.95,  # VaR confidence level
    'max_var_exposure': 0.1,  # Maximum VaR exposure
    'position_sizing': {
        'risk_per_trade': 0.02,  # Risk per trade
        'atr_multiplier': 1.5,  # ATR multiplier for position sizing
        'fixed_sl_points': None,  # Dynamic SL based on ATR
        'min_risk_reward': 1.5,  # Minimum risk/reward ratio
        'max_position_size': 1.0,  # Maximum position size in lots
    },
    'max_positions': 5,  # Maximum number of concurrent positions
    'max_drawdown': 0.10,  # Maximum allowed drawdown (10%)
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'weight': 0.3
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'weight': 0.3
    },
    'atr': {
        'period': 14,
        'weight': 0.2
    },
    'bollinger': {
        'period': 20,
        'std_dev': 2,
        'weight': 0.2
    },
    'momentum': {
        'roc_period': 10,
        'cci_period': 20,
        'willr_period': 14,
        'weight': 0.2
    },
    'volume': {
        'period': 20,
        'weight': 0.1
    }
}

# Market Analysis
MARKET_PARAMS = {
    'min_daily_volume': 1000,  # Minimum daily volume
    'max_spread_percent': 0.1,  # Maximum spread as percentage of price
    'orderbook_depth': 10,  # Order book analysis depth
    'volatility_threshold': 0.02,  # Maximum allowed volatility
}

# Performance Monitoring
PERFORMANCE_PARAMS = {
    'metrics': ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate'],
    'benchmark': 'SPY',  # Benchmark for performance comparison
    'rolling_window': 20,  # Rolling window for metrics calculation
}

# Telegram Configuration
TELEGRAM_CONFIG = {
    'enabled': True,
    'token': None,  # Will be set from environment
    'chat_id': None,  # Will be set from environment
    'notification_types': {
        'trades': True,
        'signals': True,
        'errors': True,
        'performance': True
    }
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 3333,
    'update_interval': 5,  # Update interval in seconds
    'charts': ['equity_curve', 'drawdown', 'positions', 'signals'],
    'show_orderbook': True,
    'show_correlation_matrix': True
}

# System Configuration
SYSTEM_PARAMS = {
    'debug_mode': False,
    'log_level': 'INFO',
    'save_trades': True,
    'backup_interval': 3600,  # Backup interval in seconds
    'recovery_mode': True,  # Auto recovery from errors
    'max_retries': 3,  # Maximum retries on error
}
