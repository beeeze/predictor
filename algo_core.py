import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta

def get_low_cap_stocks(max_market_cap=1e9, limit=100):
    """
    Returns a list of low-cap stocks under a given market cap
    """
    ticker_list = [
        "GME", "AMC", "BB", "NOK", "PLTR", "RKT", "CLSK", "IPOF", "DDOG", "SOUN",
        "RKLB", "BARK", "EXPR", "ZOM", "AEO", "LWAY", "CRSR", "HIMS", "BYND", "TLRY",
        "CGC", "APHA", "MTCH", "PST", "GGPI", "FTXN", "DTC", "WSTG", "TMBR", "UWMC"
    ]
    
    low_cap = []
    for ticker in ticker_list:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'marketCap' in info and info['marketCap'] is not None:
                if info['marketCap'] < max_market_cap:
                    low_cap.append(ticker)
        except Exception as e:
            continue  # Skip any problematic tickers
    
    return low_cap[:limit]


def fetch_data(symbol, period="6mo"):
    """
    Fetch historical data and add technical indicators
    """
    try:
        df = yf.download(symbol, period=period, interval="1d")
        df.dropna(inplace=True)
        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df.dropna()
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")


def train_model(df):
    """
    Train a model on historical data
    """
    features = df.columns[7:-1]  # Skip OHLCV and select TA features
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model, features


def predict_jump(model, features, symbol):
    """
    Predict probability of next-day price jump
    """
    try:
        df = fetch_data(symbol)
        latest_row = df.iloc[-1][features].values.reshape(1, -1)
        proba = model.predict_proba(latest_row)[0][1]  # Probability of rising
        return proba
    except Exception as e:
        return 0  # Return 0 probability if prediction fails


def scan_for_opportunities(threshold=0.55, max_cap=1e9):
    """
    Scan all low-cap stocks and return those with high probability to rise
    """
    candidates = get_low_cap_stocks(max_cap)
    results = []

    for symbol in candidates:
        try:
            df = fetch_data(symbol)
            if len(df) < 10:
                continue

            model, features = train_model(df)
            probability = predict_jump(model, features, symbol)

            if probability >= threshold:
                results.append({
                    'symbol': symbol,
                    'probability': round(probability * 100, 2)
                })
        except Exception as e:
            continue  # Skip any problematic tickers

    if not results:
        return pd.DataFrame()  # Return empty DataFrame if no results found

    return pd.DataFrame(results).sort_values(by='probability', ascending=False)
