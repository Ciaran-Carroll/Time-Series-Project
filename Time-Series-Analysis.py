import yfinance as yf
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import sys
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def download_stock_data(ticker, start="2010-01-01", end=None, filename=None, save_csv=True):
    """
    Downloads historical stock price data for a given ticker symbol.
    Parameters:
        ticker (str): Stock ticker (e.g., 'AAPL', 'GOOGL')
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format (None = today)
        filename (str): Output CSV filename (default = '<ticker>.csv')
        save_csv (bool): Whether to save data to CSV (default = True)

    Features:
        Handles errors
        Checks for empty data
        Optionally saves to csv

    Returns:
        DataFrame: Historical data or None if error
    """
    print(f"Downloading data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if data.empty:
        print("No data found. Check the ticker symbol.")
        return None

    print(f"Downloaded {len(data)} rows from {data.index[0].date()} to {data.index[-1].date()}")

    if save_csv:
        if not filename:
            filename = f"{ticker}.csv"
        data.to_csv(filename)
        print(f"Saved dataset to {filename}")

    return data


## Time Series Analysis Functions

def calculate_moving_averages(data, windows=[20, 50, 200]):
    """
    Calculate simple moving averages for specified windows.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        windows (list): List of window sizes in days

    Returns:
        DataFrame: Original data with MA columns added
    """
    data = data.copy()
    for window in windows:
        data[f'MA{window}'] = data['Close'].rolling(window=window).mean()
    return data


def calculate_returns(data, periods=[1, 7, 30]):
    """
    Calculate daily returns and returns over specified periods.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        periods (list): List of periods in days

    Returns:
        DataFrame: Original data with returns columns added
    """
    data = data.copy()
    data['Daily_Return'] = data['Close'].pct_change()

    for period in periods:
        data[f'Return_{period}d'] = data['Close'].pct_change(periods=period)

    return data


def calculate_volatility(data, window=30):
    """
    Calculate rolling volatility (standard deviation of returns).

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        window (int): Rolling window size in days

    Returns:
        DataFrame: Original data with volatility column added
    """
    data = data.copy()
    if 'Daily_Return' not in data.columns:
        data['Daily_Return'] = data['Close'].pct_change()

    data[f'Volatility_{window}d'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
    return data


def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        window (int): RSI period (typically 14 days)

    Returns:
        DataFrame: Original data with RSI column added
    """
    data = data.copy()
    delta = data['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        window (int): Moving average period
        num_std (int): Number of standard deviations

    Returns:
        DataFrame: Original data with Bollinger Band columns added
    """
    data = data.copy()
    data['BB_Middle'] = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + (rolling_std * num_std)
    data['BB_Lower'] = data['BB_Middle'] - (rolling_std * num_std)
    return data


def detect_trend(data, short_window=50, long_window=200):
    """
    Detect trend using moving average crossover strategy.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        short_window (int): Short-term MA window
        long_window (int): Long-term MA window

    Returns:
        DataFrame: Original data with trend signal column
    """
    data = data.copy()
    data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window).mean()

    # 1 = bullish (short MA > long MA), -1 = bearish, 0 = no signal
    data['Trend_Signal'] = 0
    data.loc[data['MA_Short'] > data['MA_Long'], 'Trend_Signal'] = 1
    data.loc[data['MA_Short'] < data['MA_Long'], 'Trend_Signal'] = -1

    return data


def apply_all_timeseries_analysis(data):
    """
    Apply all time series analysis functions to the data.

    Parameters:
        data (DataFrame): Stock data with 'Close' column

    Returns:
        DataFrame: Data with all time series indicators
    """
    data = calculate_moving_averages(data)
    data = calculate_returns(data)
    data = calculate_volatility(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = detect_trend(data)
    return data


def print_timeseries_summary(data):
    """
    Print summary statistics of time series analysis.

    Parameters:
        data (DataFrame): Stock data with time series indicators
    """
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("="*60)

    if 'Daily_Return' in data.columns:
        print(f"\nReturns Analysis:")
        print(f"  Average Daily Return: {data['Daily_Return'].mean()*100:.4f}%")
        print(f"  Daily Return Std Dev: {data['Daily_Return'].std()*100:.4f}%")
        print(f"  Total Return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1)*100:.2f}%")

    if 'Volatility_30d' in data.columns:
        print(f"\nVolatility (30-day):")
        print(f"  Current: {data['Volatility_30d'].iloc[-1]*100:.2f}%")
        print(f"  Average: {data['Volatility_30d'].mean()*100:.2f}%")

    if 'RSI' in data.columns:
        current_rsi = data['RSI'].iloc[-1]
        print(f"\nRSI (14-day):")
        print(f"  Current: {current_rsi:.2f}")
        if current_rsi > 70:
            print(f"  Signal: Overbought")
        elif current_rsi < 30:
            print(f"  Signal: Oversold")
        else:
            print(f"  Signal: Neutral")

    if 'Trend_Signal' in data.columns:
        current_trend = data['Trend_Signal'].iloc[-1]
        print(f"\nTrend (50/200 MA Crossover):")
        if current_trend == 1:
            print(f"  Current Signal: Bullish (Golden Cross)")
        elif current_trend == -1:
            print(f"  Current Signal: Bearish (Death Cross)")
        else:
            print(f"  Current Signal: Neutral")

    print("="*60 + "\n")


## Stock Data Visualisation

def plot_stock_data(data, ticker):
    """
    Plots the Close price and Volume of the stock dataset.
    Includes properly formatted date axes to provide immediate insight
    into price trends and trading activity.

    Parameters:
        data (DataFrame): Stock data with 'Close' and 'Volume' columns
        ticker (str): Stock ticker symbol for the title

    Features:
        Two subplots: closing price (top) and volume (bottom)
        Currency formatting on y-axis ($)
        Volume displayed in millions (M)
        Date formatting on x-axis (YYYY-MM)
        Rotated date labels for readability
        Color-coded and styled lines
        Detailed dataset info printed
        Professional appearance with grid and proper sizing
    """
    if data is None or data.empty:
        print("No data to plot.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Plot closing price
    ax1.plot(data.index, data["Close"], color='#1f77b4', linewidth=2)
    ax1.set_title(f"{ticker} Closing Price Over Time", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Closing Price (USD)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Format y-axis as currency
    def currency(x, pos):
        return f'${x:,.0f}'
    ax1.yaxis.set_major_formatter(FuncFormatter(currency))

    # Plot volume
    ax2.bar(data.index, data["Volume"], color='gray', alpha=0.5, width=1)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format volume axis
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # Rotate date labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Print dataset info
    print(f"Dataset contains {len(data)} days from {data.index[0].date()} to {data.index[-1].date()}")


def plot_timeseries_analysis(data, ticker):
    """
    Plot comprehensive time series analysis including MAs, RSI, and Bollinger Bands.

    Parameters:
        data (DataFrame): Stock data with time series indicators
        ticker (str): Stock ticker symbol
    """
    if data is None or data.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1, 1]})

    # Plot 1: Price with Moving Averages and Bollinger Bands
    ax1 = axes[0]
    ax1.plot(data.index, data['Close'], label='Close', color='black', linewidth=2)

    if 'MA20' in data.columns:
        ax1.plot(data.index, data['MA20'], label='MA20', color='blue', linewidth=1.5, alpha=0.7)
    if 'MA50' in data.columns:
        ax1.plot(data.index, data['MA50'], label='MA50', color='orange', linewidth=1.5, alpha=0.7)
    if 'MA200' in data.columns:
        ax1.plot(data.index, data['MA200'], label='MA200', color='red', linewidth=1.5, alpha=0.7)

    if 'BB_Upper' in data.columns:
        ax1.plot(data.index, data['BB_Upper'], label='BB Upper', color='gray',
                linestyle='--', linewidth=1, alpha=0.5)
        ax1.plot(data.index, data['BB_Lower'], label='BB Lower', color='gray',
                linestyle='--', linewidth=1, alpha=0.5)
        ax1.fill_between(data.index, data['BB_Lower'], data['BB_Upper'],
                         color='gray', alpha=0.1)

    ax1.set_title(f"{ticker} - Price with Moving Averages & Bollinger Bands",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))

    # Plot 2: RSI
    ax2 = axes[1]
    if 'RSI' in data.columns:
        ax2.plot(data.index, data['RSI'], label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Oversold')
        ax2.fill_between(data.index, 30, 70, color='gray', alpha=0.1)
        ax2.set_ylabel("RSI", fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Volume
    ax3 = axes[2]
    ax3.bar(data.index, data['Volume'], color='steelblue', alpha=0.5, width=1)
    ax3.set_ylabel("Volume", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M'))
    ax3.grid(True, alpha=0.3)

    # Format x-axis dates for all plots
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


## ARIMA and SARIMA Models

def check_stationarity(series, name="Series"):
    """
    Perform Augmented Dickey-Fuller test to check stationarity.

    Parameters:
        series (Series): Time series data
        name (str): Name of the series for display

    Returns:
        bool: True if stationary, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STATIONARITY TEST: {name}")
    print('='*60)

    # Remove NaN values
    series_clean = series.dropna()

    result = adfuller(series_clean)
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.3f}")

    is_stationary = result[1] < 0.05
    if is_stationary:
        print(f"\nResult: Series is STATIONARY (p-value < 0.05)")
    else:
        print(f"\nResult: Series is NON-STATIONARY (p-value >= 0.05)")
        print("Recommendation: Consider differencing the series")

    return is_stationary


def plot_acf_pacf(series, lags=40):
    """
    Plot ACF and PACF to help identify ARIMA parameters.

    Parameters:
        series (Series): Time series data
        lags (int): Number of lags to display
    """
    series_clean = series.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(series_clean, lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(series_clean, lags=lags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation Function (PACF)")

    plt.tight_layout()
    plt.show()


def fit_arima_model(data, order=(1, 1, 1), forecast_steps=30):
    """
    Fit an ARIMA model to the closing price data.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        order (tuple): ARIMA order (p, d, q)
        forecast_steps (int): Number of steps to forecast

    Returns:
        dict: Dictionary containing model, forecast, and metrics
    """
    print(f"\n{'='*60}")
    print(f"FITTING ARIMA{order} MODEL")
    print('='*60)

    # Prepare data
    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Fit model
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()

        print(f"\nModel Summary:")
        print(f"AIC: {model_fit.aic:.2f}")
        print(f"BIC: {model_fit.bic:.2f}")

        # Make predictions on test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        print(f"\nTest Set Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Forecast future
        future_forecast = model_fit.forecast(steps=forecast_steps)

        return {
            'model': model_fit,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': future_forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': model_fit.aic, 'BIC': model_fit.bic},
            'order': order
        }

    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None


def fit_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), forecast_steps=30):
    """
    Fit a SARIMA model to the closing price data.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        order (tuple): ARIMA order (p, d, q)
        seasonal_order (tuple): Seasonal order (P, D, Q, s)
        forecast_steps (int): Number of steps to forecast

    Returns:
        dict: Dictionary containing model, forecast, and metrics
    """
    print(f"\n{'='*60}")
    print(f"FITTING SARIMA{order}x{seasonal_order} MODEL")
    print('='*60)

    # Prepare data
    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Fit model
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)

        print(f"\nModel Summary:")
        print(f"AIC: {model_fit.aic:.2f}")
        print(f"BIC: {model_fit.bic:.2f}")

        # Make predictions on test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        print(f"\nTest Set Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Forecast future
        future_forecast = model_fit.forecast(steps=forecast_steps)

        return {
            'model': model_fit,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': future_forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': model_fit.aic, 'BIC': model_fit.bic},
            'order': order,
            'seasonal_order': seasonal_order
        }

    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        return None


def compare_models(arima_result, sarima_result):
    """
    Compare ARIMA and SARIMA model performance.

    Parameters:
        arima_result (dict): Results from fit_arima_model
        sarima_result (dict): Results from fit_sarima_model
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print('='*60)

    if arima_result is None or sarima_result is None:
        print("Cannot compare: one or both models failed to fit")
        return

    # Create comparison table
    metrics = ['RMSE', 'MAE', 'MAPE', 'AIC', 'BIC']

    print(f"\n{'Metric':<10} {'ARIMA':<15} {'SARIMA':<15} {'Winner':<10}")
    print('-' * 60)

    for metric in metrics:
        arima_val = arima_result['metrics'][metric]
        sarima_val = sarima_result['metrics'][metric]

        # Lower is better for all these metrics
        winner = "ARIMA" if arima_val < sarima_val else "SARIMA"

        if metric == 'MAPE':
            print(f"{metric:<10} {arima_val:<15.2f} {sarima_val:<15.2f} {winner:<10}")
        else:
            print(f"{metric:<10} {arima_val:<15.2f} {sarima_val:<15.2f} {winner:<10}")

    print('=' * 60)


def plot_model_results(data, arima_result, sarima_result, ticker):
    """
    Plot actual vs predicted values for both models.

    Parameters:
        data (DataFrame): Original stock data
        arima_result (dict): Results from fit_arima_model
        sarima_result (dict): Results from fit_sarima_model
        ticker (str): Stock ticker symbol
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ARIMA Plot
    if arima_result is not None:
        ax1 = axes[0]

        # Plot training data
        ax1.plot(arima_result['train'].index, arima_result['train'],
                label='Training Data', color='blue', linewidth=1.5)

        # Plot test data
        ax1.plot(arima_result['test'].index, arima_result['test'],
                label='Actual Test Data', color='green', linewidth=1.5)

        # Plot predictions
        ax1.plot(arima_result['test'].index, arima_result['predictions'],
                label='ARIMA Predictions', color='red', linewidth=1.5, linestyle='--')

        # Plot forecast
        last_date = arima_result['test'].index[-1]
        forecast_index = pd.date_range(start=last_date, periods=len(arima_result['forecast'])+1, freq='D')[1:]
        ax1.plot(forecast_index, arima_result['forecast'],
                label='Future Forecast', color='orange', linewidth=2, linestyle=':')

        ax1.set_title(f"{ticker} - ARIMA{arima_result['order']} Model",
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price (USD)", fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))

    # SARIMA Plot
    if sarima_result is not None:
        ax2 = axes[1]

        # Plot training data
        ax2.plot(sarima_result['train'].index, sarima_result['train'],
                label='Training Data', color='blue', linewidth=1.5)

        # Plot test data
        ax2.plot(sarima_result['test'].index, sarima_result['test'],
                label='Actual Test Data', color='green', linewidth=1.5)

        # Plot predictions
        ax2.plot(sarima_result['test'].index, sarima_result['predictions'],
                label='SARIMA Predictions', color='red', linewidth=1.5, linestyle='--')

        # Plot forecast
        last_date = sarima_result['test'].index[-1]
        forecast_index = pd.date_range(start=last_date, periods=len(sarima_result['forecast'])+1, freq='D')[1:]
        ax2.plot(forecast_index, sarima_result['forecast'],
                label='Future Forecast', color='orange', linewidth=2, linestyle=':')

        ax2.set_title(f"{ticker} - SARIMA{sarima_result['order']}x{sarima_result['seasonal_order']} Model",
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel("Price (USD)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))

    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def auto_arima_selection(data, max_p=5, max_d=2, max_q=5):
    """
    Automatically select best ARIMA parameters using grid search based on AIC.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        max_p (int): Maximum AR order
        max_d (int): Maximum differencing order
        max_q (int): Maximum MA order

    Returns:
        tuple: Best (p, d, q) order
    """
    print(f"\n{'='*60}")
    print("AUTO ARIMA PARAMETER SELECTION")
    print('='*60)
    print("Searching for optimal parameters... This may take a while.")

    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train = series[:train_size]

    best_aic = np.inf
    best_order = None
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    results.append(((p, d, q), aic))

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except:
                    continue

    # Display top 5 models
    results.sort(key=lambda x: x[1])
    print(f"\nTop 5 Models by AIC:")
    print(f"{'Order':<15} {'AIC':<15}")
    print('-' * 30)
    for i, (order, aic) in enumerate(results[:5], 1):
        marker = "***" if i == 1 else ""
        print(f"{str(order):<15} {aic:<15.2f} {marker}")

    print(f"\nBest ARIMA order: {best_order} with AIC: {best_aic:.2f}")
    return best_order

def fit_ses_model(data, forecast_steps=30, smoothing_level=None):
    """
    Fit Simple Exponential Smoothing model to closing price data.
    Best for data without trend or seasonality.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        forecast_steps (int): Number of steps to forecast
        smoothing_level (float): Alpha parameter (0-1), None for auto-optimization

    Returns:
        dict: Dictionary containing model, forecast, and metrics
    """
    print(f"\n{'='*60}")
    print(f"FITTING SIMPLE EXPONENTIAL SMOOTHING (SES) MODEL")
    print('='*60)

    # Prepare data
    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Fit model
    try:
        if smoothing_level is not None:
            model = SimpleExpSmoothing(train)
            model_fit = model.fit(smoothing_level=smoothing_level)
            print(f"Smoothing level (alpha): {smoothing_level:.4f} (user-specified)")
        else:
            model = SimpleExpSmoothing(train)
            model_fit = model.fit(optimized=True)
            print(f"Smoothing level (alpha): {model_fit.params['smoothing_level']:.4f} (optimized)")

        print(f"AIC: {model_fit.aic:.2f}")

        # Make predictions on test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        print(f"\nTest Set Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Forecast future
        future_forecast = model_fit.forecast(steps=forecast_steps)

        return {
            'model': model_fit,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': future_forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': model_fit.aic},
            'params': model_fit.params,
            'name': 'SES'
        }

    except Exception as e:
        print(f"Error fitting SES model: {e}")
        return None


def fit_holt_model(data, forecast_steps=30, smoothing_level=None, smoothing_trend=None, damped_trend=False):
    """
    Fit Holt's Linear Trend model to closing price data.
    Handles trend but not seasonality.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        forecast_steps (int): Number of steps to forecast
        smoothing_level (float): Alpha parameter (0-1), None for auto-optimization
        smoothing_trend (float): Beta parameter (0-1), None for auto-optimization
        damped_trend (bool): Whether to use damped trend

    Returns:
        dict: Dictionary containing model, forecast, and metrics
    """
    print(f"\n{'='*60}")
    print(f"FITTING HOLT'S LINEAR TREND MODEL")
    if damped_trend:
        print("(with damped trend)")
    print('='*60)

    # Prepare data
    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Fit model
    try:
        model = Holt(train, damped_trend=damped_trend)

        if smoothing_level is not None and smoothing_trend is not None:
            model_fit = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)
            print(f"Smoothing level (alpha): {smoothing_level:.4f} (user-specified)")
            print(f"Smoothing trend (beta): {smoothing_trend:.4f} (user-specified)")
        else:
            model_fit = model.fit(optimized=True)
            print(f"Smoothing level (alpha): {model_fit.params['smoothing_level']:.4f} (optimized)")
            print(f"Smoothing trend (beta): {model_fit.params['smoothing_trend']:.4f} (optimized)")

        if damped_trend:
            print(f"Damping parameter (phi): {model_fit.params.get('damping_trend', 'N/A'):.4f}")

        print(f"AIC: {model_fit.aic:.2f}")

        # Make predictions on test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        print(f"\nTest Set Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Forecast future
        future_forecast = model_fit.forecast(steps=forecast_steps)

        return {
            'model': model_fit,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': future_forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': model_fit.aic},
            'params': model_fit.params,
            'name': 'Holt (Damped)' if damped_trend else 'Holt'
        }

    except Exception as e:
        print(f"Error fitting Holt model: {e}")
        return None


def fit_holtwinters_model(data, seasonal_periods=12, trend='add', seasonal='add',
                         forecast_steps=30, damped_trend=False):
    """
    Fit Holt-Winters (Triple Exponential Smoothing) model to closing price data.
    Handles both trend and seasonality.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        seasonal_periods (int): Number of periods in season (e.g., 12 for monthly, 252 for daily yearly)
        trend (str): Type of trend component ('add', 'mul', or None)
        seasonal (str): Type of seasonal component ('add', 'mul', or None)
        forecast_steps (int): Number of steps to forecast
        damped_trend (bool): Whether to use damped trend

    Returns:
        dict: Dictionary containing model, forecast, and metrics
    """
    print(f"\n{'='*60}")
    print(f"FITTING HOLT-WINTERS (TRIPLE EXPONENTIAL SMOOTHING) MODEL")
    print(f"Trend: {trend}, Seasonal: {seasonal}, Periods: {seasonal_periods}")
    if damped_trend:
        print("(with damped trend)")
    print('='*60)

    # Prepare data
    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]

    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")

    # Check if we have enough data for the seasonal period
    if len(train) < 2 * seasonal_periods:
        print(f"Warning: Training data ({len(train)} points) may be insufficient for seasonal_periods={seasonal_periods}")
        print("Recommendation: Use a smaller seasonal_periods or more data")

    # Fit model
    try:
        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend
        )
        model_fit = model.fit(optimized=True)

        print(f"\nOptimized Parameters:")
        print(f"Smoothing level (alpha): {model_fit.params['smoothing_level']:.4f}")
        if trend:
            print(f"Smoothing trend (beta): {model_fit.params['smoothing_trend']:.4f}")
        if seasonal:
            print(f"Smoothing seasonal (gamma): {model_fit.params['smoothing_seasonal']:.4f}")
        if damped_trend:
            print(f"Damping parameter (phi): {model_fit.params.get('damping_trend', 'N/A'):.4f}")

        print(f"AIC: {model_fit.aic:.2f}")

        # Make predictions on test set
        predictions = model_fit.forecast(steps=len(test))

        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100

        print(f"\nTest Set Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        # Forecast future
        future_forecast = model_fit.forecast(steps=forecast_steps)

        return {
            'model': model_fit,
            'train': train,
            'test': test,
            'predictions': predictions,
            'forecast': future_forecast,
            'metrics': {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'AIC': model_fit.aic},
            'params': model_fit.params,
            'seasonal_periods': seasonal_periods,
            'trend': trend,
            'seasonal': seasonal,
            'name': 'Holt-Winters'
        }

    except Exception as e:
        print(f"Error fitting Holt-Winters model: {e}")
        return None


def compare_all_models(*model_results):
    """
    Compare multiple forecasting models (ARIMA, SARIMA, SES, Holt, Holt-Winters).

    Parameters:
        *model_results: Variable number of model result dictionaries
    """
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print('='*60)

    # Filter out None results
    valid_models = [m for m in model_results if m is not None]

    if len(valid_models) < 2:
        print("Need at least 2 valid models to compare")
        return

    # Get model names
    model_names = []
    for model in valid_models:
        if 'name' in model:
            model_names.append(model['name'])
        elif 'seasonal_order' in model:
            model_names.append(f"SARIMA{model['order']}x{model['seasonal_order']}")
        else:
            model_names.append(f"ARIMA{model['order']}")

    # Create comparison table
    metrics = ['RMSE', 'MAE', 'MAPE', 'AIC']

    print(f"\n{'Metric':<10}", end='')
    for name in model_names:
        print(f"{name:<20}", end='')
    print("Winner")
    print('-' * (10 + 20 * len(model_names) + 20))

    for metric in metrics:
        print(f"{metric:<10}", end='')
        values = []

        for model in valid_models:
            if metric in model['metrics']:
                val = model['metrics'][metric]
                values.append(val)
                if metric == 'MAPE':
                    print(f"{val:<20.2f}", end='')
                else:
                    print(f"{val:<20.2f}", end='')
            else:
                values.append(float('inf'))
                print(f"{'N/A':<20}", end='')

        # Find winner (lowest value)
        winner_idx = values.index(min(values))
        print(f"{model_names[winner_idx]:<20}")

    print('=' * (10 + 20 * len(model_names) + 20))

    # Summary
    print(f"\nModel Performance Summary:")
    for i, (name, model) in enumerate(zip(model_names, valid_models)):
        print(f"{i+1}. {name}")
        print(f"   RMSE: ${model['metrics']['RMSE']:.2f}, MAE: ${model['metrics']['MAE']:.2f}, MAPE: {model['metrics']['MAPE']:.2f}%")


def plot_exponential_smoothing_results(data, ses_result, holt_result, hw_result, ticker):
    """
    Plot actual vs predicted values for exponential smoothing models.

    Parameters:
        data (DataFrame): Original stock data
        ses_result (dict): Results from fit_ses_model
        holt_result (dict): Results from fit_holt_model
        hw_result (dict): Results from fit_holtwinters_model
        ticker (str): Stock ticker symbol
    """
    # Count valid models
    valid_models = [m for m in [ses_result, holt_result, hw_result] if m is not None]
    n_models = len(valid_models)

    if n_models == 0:
        print("No models to plot")
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5*n_models))

    # Make axes iterable even if only one model
    if n_models == 1:
        axes = [axes]

    model_results = [
        (ses_result, 'SES'),
        (holt_result, 'Holt'),
        (hw_result, 'Holt-Winters')
    ]

    plot_idx = 0
    for result, name in model_results:
        if result is None:
            continue

        ax = axes[plot_idx]

        # Plot training data
        ax.plot(result['train'].index, result['train'],
                label='Training Data', color='blue', linewidth=1.5)

        # Plot test data
        ax.plot(result['test'].index, result['test'],
                label='Actual Test Data', color='green', linewidth=1.5)

        # Plot predictions
        ax.plot(result['test'].index, result['predictions'],
                label=f'{name} Predictions', color='red', linewidth=1.5, linestyle='--')

        # Plot forecast
        last_date = result['test'].index[-1]
        forecast_index = pd.date_range(start=last_date, periods=len(result['forecast'])+1, freq='D')[1:]
        ax.plot(forecast_index, result['forecast'],
                label='Future Forecast', color='orange', linewidth=2, linestyle=':')

        # Title with metrics
        title = f"{ticker} - {name} Model (RMSE: ${result['metrics']['RMSE']:.2f}, MAPE: {result['metrics']['MAPE']:.2f}%)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Price (USD)", fontsize=11)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plot_idx += 1

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.show()


def auto_holtwinters_selection(data, max_seasonal_periods=30):
    """
    Automatically select best Holt-Winters configuration using grid search.

    Parameters:
        data (DataFrame): Stock data with 'Close' column
        max_seasonal_periods (int): Maximum seasonal period to test

    Returns:
        dict: Best configuration with parameters
    """
    print(f"\n{'='*60}")
    print("AUTO HOLT-WINTERS PARAMETER SELECTION")
    print('='*60)
    print("Searching for optimal configuration... This may take a while.")

    series = data['Close'].dropna()
    train_size = int(len(series) * 0.8)
    train = series[:train_size]

    best_aic = np.inf
    best_config = None
    results = []

    # Test different configurations
    trends = ['add', 'mul', None]
    seasonals = ['add', 'mul', None]
    periods = [7, 12, 21, 30, 60, 90, 126, 252]  # Common periods for stocks

    # Filter periods that have enough data
    valid_periods = [p for p in periods if len(train) >= 2 * p and p <= max_seasonal_periods]

    for trend in trends:
        for seasonal in seasonals:
            if seasonal is None:
                # No seasonality, test once
                try:
                    model = ExponentialSmoothing(train, trend=trend, seasonal=None)
                    model_fit = model.fit(optimized=True)
                    aic = model_fit.aic
                    config = {'trend': trend, 'seasonal': None, 'periods': None}
                    results.append((config, aic))

                    if aic < best_aic:
                        best_aic = aic
                        best_config = config
                except:
                    continue
            else:
                # Test different seasonal periods
                for period in valid_periods:
                    try:
                        model = ExponentialSmoothing(
                            train,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=period
                        )
                        model_fit = model.fit(optimized=True)
                        aic = model_fit.aic
                        config = {'trend': trend, 'seasonal': seasonal, 'periods': period}
                        results.append((config, aic))

                        if aic < best_aic:
                            best_aic = aic
                            best_config = config
                    except:
                        continue

    # Display top 5 configurations
    results.sort(key=lambda x: x[1])
    print(f"\nTop 5 Configurations by AIC:")
    print(f"{'Trend':<10} {'Seasonal':<10} {'Periods':<10} {'AIC':<15}")
    print('-' * 50)
    for i, (config, aic) in enumerate(results[:5], 1):
        marker = "***" if i == 1 else ""
        trend_str = str(config['trend']) if config['trend'] else 'None'
        seasonal_str = str(config['seasonal']) if config['seasonal'] else 'None'
        periods_str = str(config['periods']) if config['periods'] else 'N/A'
        print(f"{trend_str:<10} {seasonal_str:<10} {periods_str:<10} {aic:<15.2f} {marker}")

    print(f"\nBest configuration: {best_config} with AIC: {best_aic:.2f}")
    return best_config



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_stock.py <TICKER> [START] [END] [--arima] [--sarima]")
        print("Example: python download_stock.py AAPL 2015-01-01 2024-12-31 --arima --sarima")
        print("\nOptions:")
        print("  --arima     Fit and compare ARIMA models")
        print("  --sarima    Fit and compare SARIMA models")
        print("  --auto      Automatically select best ARIMA parameters")

        if run_smooth or auto_smooth:
            ses_result = None
            holt_result = None
            hw_result = None

        if auto_smooth:
            # Auto-select best Holt-Winters configuration
            best_config = auto_holtwinters_selection(data, max_seasonal_periods=90)
            hw_result = fit_holtwinters_model(
                data,
                seasonal_periods=best_config['periods'] if best_config['periods'] else 12,
                trend=best_config['trend'],
                seasonal=best_config['seasonal'],
                forecast_steps=30
            )
        else:
            # Fit all three exponential smoothing models
            ses_result = fit_ses_model(data, forecast_steps=30)
            holt_result = fit_holt_model(data, forecast_steps=30, damped_trend=False)
            hw_result = fit_holtwinters_model(
                data,
                seasonal_periods=252,  # Annual seasonality for daily data
                trend='add',
                seasonal='add',
                forecast_steps=30
            )

        # Plot results
        plot_exponential_smoothing_results(data, ses_result, holt_result, hw_result, ticker)

        # Compare all models if multiple exist
        all_models = [m for m in [arima_result, sarima_result, ses_result, holt_result, hw_result] if m is not None]
        if len(all_models) >= 2:
            compare_all_models(*all_models)
        sys.exit(1)

    ticker = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "2010-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None

    # Check for flags
    run_arima = '--arima' in sys.argv
    run_sarima = '--sarima' in sys.argv
    auto_select = '--auto' in sys.argv
    run_smooth = '--smooth' in sys.argv
    auto_smooth = '--auto-smooth' in sys.argv

    data = download_stock_data(ticker, start, end)

    if data is not None:
        # Apply time series analysis
        data = apply_all_timeseries_analysis(data)

        # Display results
        plot_stock_data(data, ticker)
        plot_timeseries_analysis(data, ticker)
        print_timeseries_summary(data)

        # ARIMA/SARIMA modeling if requested
        if run_arima or run_sarima or auto_select:
            # Check stationarity
            check_stationarity(data['Close'], name="Closing Price")

            # Check differenced series
            diff_series = data['Close'].diff().dropna()
            check_stationarity(diff_series, name="Differenced Closing Price")

            # Plot ACF and PACF
            print("\nPlotting ACF and PACF for differenced series...")
            plot_acf_pacf(diff_series, lags=40)

            arima_result = None
            sarima_result = None

            if auto_select:
                # Auto-select best parameters
                best_order = auto_arima_selection(data, max_p=3, max_d=2, max_q=3)
                arima_result = fit_arima_model(data, order=best_order, forecast_steps=30)
            elif run_arima:
                # Fit ARIMA with default or custom parameters
                arima_result = fit_arima_model(data, order=(1, 1, 1), forecast_steps=30)

            if run_sarima:
                # Fit SARIMA model
                sarima_result = fit_sarima_model(data, order=(1, 1, 1),
                                                seasonal_order=(1, 1, 1, 12),
                                                forecast_steps=30)

            # Compare models if both were fitted
            if arima_result is not None and sarima_result is not None:
                compare_models(arima_result, sarima_result)
                plot_model_results(data, arima_result, sarima_result, ticker)
            elif arima_result is not None:
                # Plot only ARIMA
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(arima_result['train'].index, arima_result['train'],
                       label='Training Data', color='blue', linewidth=1.5)
                ax.plot(arima_result['test'].index, arima_result['test'],
                       label='Actual Test Data', color='green', linewidth=1.5)
                ax.plot(arima_result['test'].index, arima_result['predictions'],
                       label='ARIMA Predictions', color='red', linewidth=1.5, linestyle='--')

                last_date = arima_result['test'].index[-1]
                forecast_index = pd.date_range(start=last_date,
                                              periods=len(arima_result['forecast'])+1, freq='D')[1:]
                ax.plot(forecast_index, arima_result['forecast'],
                       label='Future Forecast', color='orange', linewidth=2, linestyle=':')

                ax.set_title(f"{ticker} - ARIMA{arima_result['order']} Model",
                           fontsize=14, fontweight='bold')
                ax.set_ylabel("Price (USD)", fontsize=11)
                ax.set_xlabel("Date", fontsize=11)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            elif sarima_result is not None:
                # Plot only SARIMA
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(sarima_result['train'].index, sarima_result['train'],
                       label='Training Data', color='blue', linewidth=1.5)
                ax.plot(sarima_result['test'].index, sarima_result['test'],
                       label='Actual Test Data', color='green', linewidth=1.5)
                ax.plot(sarima_result['test'].index, sarima_result['predictions'],
                       label='SARIMA Predictions', color='red', linewidth=1.5, linestyle='--')

                last_date = sarima_result['test'].index[-1]
                forecast_index = pd.date_range(start=last_date,
                                              periods=len(sarima_result['forecast'])+1, freq='D')[1:]
                ax.plot(forecast_index, sarima_result['forecast'],
                       label='Future Forecast', color='orange', linewidth=2, linestyle=':')

                ax.set_title(f"{ticker} - SARIMA{sarima_result['order']}x{sarima_result['seasonal_order']} Model",
                           fontsize=14, fontweight='bold')
                ax.set_ylabel("Price (USD)", fontsize=11)
                ax.set_xlabel("Date", fontsize=11)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                plt.show()


                if run_smooth or auto_smooth:
                    ses_result = None
                    holt_result = None
                    hw_result = None

                    if auto_smooth:
                        # Auto-select best Holt-Winters configuration
                        best_config = auto_holtwinters_selection(data, max_seasonal_periods=90)
                        hw_result = fit_holtwinters_model(
                        data,
                        seasonal_periods=best_config['periods'] if best_config['periods'] else 12,
                        trend=best_config['trend'],
                        seasonal=best_config['seasonal'],
                        forecast_steps=30
                        )
                    else:
                        # Fit all three exponential smoothing models
                        ses_result = fit_ses_model(data, forecast_steps=30)
                        holt_result = fit_holt_model(data, forecast_steps=30, damped_trend=False)
                        hw_result = fit_holtwinters_model(
                        data,
                        seasonal_periods=252,  # Annual seasonality for daily data
                        trend='add',
                        seasonal='add',
                        forecast_steps=30
                        )

                        # Plot results
                        plot_exponential_smoothing_results(data, ses_result, holt_result, hw_result, ticker)

                        # Compare all models if multiple exist
                        all_models = [m for m in [arima_result, sarima_result, ses_result, holt_result, hw_result] if m is not None]
                        if len(all_models) >= 2:
                            compare_all_models(*all_models)


## What the models do:

# Split data into 80% training, 20% testing
# Test for stationarity and suggest differencing if needed
# Display ACF/PACF plots to guide parameter selection
# Compare forecasts against actual test data
# Generate 30-day future forecasts
# Calculate performance metrics (RMSE, MAE, MAPE, AIC, BIC)


# What Each Flag Does:
#
# --smooth: Fits SES, Holt, and Holt-Winters models with default parameters
# --auto-smooth: Automatically finds the best Holt-Winters configuration
# --arima: Fits ARIMA model
# --sarima: Fits SARIMA model
# --auto: Auto-selects best ARIMA parameters
#
# The program will now display comprehensive comparisons across all models you select!
