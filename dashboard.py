from flask import Flask, render_template, jsonify, request
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objs as go
import threading
import time
import asyncio
import websockets
from config import MT5_CONFIG

app = Flask(__name__)

async def send_realtime_data(websocket, path):
    """Gerçek zamanlı fiyat verisini WebSocket ile gönder"""
    while True:
        symbol = "EURUSD"
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is not None:
            df = pd.DataFrame(rates)
            price_data = {
                "time": str(pd.to_datetime(df['time'][0], unit='s')),
                "open": df['open'][0],
                "high": df['high'][0],
                "low": df['low'][0],
                "close": df['close'][0]
            }
            await websocket.send(jsonify(price_data))
        await asyncio.sleep(1)

class TradingDashboard:
    def __init__(self):
        print("Dashboard initialized")

    def run(self, host='127.0.0.1', port=5000):
        print(f"Dashboard running on {host}:{port}")


def connect_mt5():
    """Bağlantıyı başlat"""
    if not mt5.initialize(
        login=MT5_CONFIG['login'],
        password=MT5_CONFIG['password'],
        server=MT5_CONFIG['server']
    ):
        return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/account_info')
def account_info():
    """Hesap bilgilerini JSON olarak döndür"""
    account = mt5.account_info()
    if account:
        return jsonify({
            "login": account.login,
            "balance": account.balance,
            "equity": account.equity,
            "profit": account.profit,
            "margin": account.margin
        })
    return jsonify({"error": "MT5 bağlantısı başarısız"})

@app.route('/get_chart')
def get_chart():
    """Canlı fiyat grafiği oluştur"""
    symbol = request.args.get('symbol', 'EURUSD')
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
    if rates is None:
        return jsonify({"error": "Veri alınamadı"})
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Fiyat"
    ))
    return jsonify(fig.to_json())

if __name__ == '__main__':
    if connect_mt5():
        server = websockets.serve(send_realtime_data, "0.0.0.0", 5678)
        asyncio.get_event_loop().run_until_complete(server)
        threading.Thread(target=lambda: app.run(debug=True, host='0.0.0.0', port=5000)).start()
    else:
        print("MT5 bağlantısı kurulamadı")
