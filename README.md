The script implements the following workflow.

1. Download data → Visualise
2. Check stationarity → Difference if needed
3. ACF/PACF → Suggest p,q values
4. Fit ARIMA/SARIMA → Compare metrics
5. Forecast → Visualise predictions

This is a comprehensive time series analysis tool for stock data built with Python.

### Overview
The code provides a complete pipeline for:
1. Downloading historical stock data
2. Performing technical analysis with common indicators
3. Implementing ARIMA/SARIMA forecasting models
4. Visualising results

### Main Components
1. **Data Download** (`download_stock_data`)
   - Uses `yfinance` to fetch stock data
   - Handles errors and empty data gracefully
   - Optionally saves to CSV
2. **Technical Analysis Functions**
   - **Moving Averages** (`calculate_moving_averages`): Calculates SMA for common windows (20, 50, 200 days)
   - **Returns** (`calculate_returns`): Computes daily and period returns
   - **Volatility** (`calculate_volatility`): Rolling standard deviation (annualised)
   - **RSI** (`calculate_rsi`): Relative Strength Index for overbought/oversold signals
   - **Bollinger Bands** (`calculate_bollinger_bands`): Price volatility bands
   - **Trend Detection** (`detect_trend`): Identifies bullish/bearish trends using MA crossovers
3. **Visualisation Functions**
   - `plot_stock_data`: Basic price and volume chart
   - `plot_timeseries_analysis`: Comprehensive view with indicators
   - Professional formatting with date axes, currency formatting, and volume scaling
4. **Time Series Modelling (ARIMA/SARIMA)**
   - **Stationarity Testing** (`check_stationarity`): Augmented Dickey-Fuller test
   - **Parameter Identification** (`plot_acf_pacf`): ACF/PACF plots for AR/MA order selection
   - **Model Fitting:**
      - `fit_arima_model`: Standard ARIMA implementation
      - `fit_sarima_model`: Seasonal ARIMA with seasonal components
   - **Auto-selection** (`auto_arima_selection`): Grid search for optimal parameters
   - **Model Comparison** (`compare_models`): Evaluates performance metrics (RMSE, MAE, MAPE, AIC, BIC)
5. **Main Execution Flow**
   When run from the command line:
   ```bash
   python Time-Series.py AAPL 2015-01-01 2024-12-31 --arima --sarima
   ```
   1. Downloads AAPL data from 2015-2024
   2. Applies all technical indicators
   3. Generates visualisations
   4. If flags specified:
      - Checks stationarity
      - Plots ACF/PACF
      - Fits ARIMA/SARIMA models
      - Compares performance
      - Generates forecasts

### Key Features
- **Modular Design:** Each function handles a specific task
- **Error Handling:** Graceful handling of download/processing errors
- **Flexible Input:** Command-line arguments with multiple options
- **Professional Output:** Well-formatted charts and summary statistics
- **Model Evaluation:** Multiple metrics for comparing forecasting performance

#### Use Cases
1. **Technical Analysis:** Calculate and visualise common trading indicators
2. **Trend Analysis:** Identify market trends using moving averages
3. **Volatility Assessment:** Measure and monitor price volatility
4. **Price Forecasting:** Predict future prices using ARIMA/SARIMA models
5. **Model Comparison:** Evaluate different forecasting approaches

#### Dependencies Required
- `yfinance`: Data download
- `pandas`, `numpy`: Data manipulation
- `matplotlib`: Visualisation
- `statsmodels`: Time series modelling
- `scikit-learn`: Model evaluation metrics

This is a production-ready toolkit for stock market analysis that combines traditional technical analysis with modern statistical forecasting methods.
