# Time-Series Analysis Program - Complete Explanation

## Overview
This is a comprehensive stock market analysis and forecasting tool that downloads historical stock data, performs technical analysis, and uses multiple forecasting models to predict future prices.

---

## Installation & Dependencies

### Required Python Version
- **Python 3.8 or higher** recommended
- The program uses f-strings and modern type hints

### Installing Dependencies

**Option 1: Using pip (recommended)**
```bash
pip install yfinance matplotlib pandas numpy statsmodels scikit-learn
```

**Option 2: Using a requirements.txt file**

Create a file named `requirements.txt` with the following content:
```
yfinance>=0.2.3
matplotlib>=3.5.0
pandas>=1.4.0
numpy>=1.21.0
statsmodels>=0.13.0
scikit-learn>=1.0.0
```

Then install with:
```bash
pip install -r requirements.txt
```

**Option 3: Using conda**
```bash
conda install -c conda-forge yfinance matplotlib pandas numpy statsmodels scikit-learn
```

### Dependency Breakdown

**1. yfinance** (`import yfinance as yf`)
- **Purpose:** Downloads historical stock data from Yahoo Finance
- **Used for:** Getting OHLCV (Open, High, Low, Close, Volume) data
- **Version:** 0.2.3 or higher recommended
- **Installation:** `pip install yfinance`

**2. matplotlib** (`from matplotlib import pyplot as plt`)
- **Purpose:** Creates all visualizations and charts
- **Used for:** Price charts, technical indicator plots, forecast visualizations
- **Submodules used:**
  - `matplotlib.pyplot` - Main plotting interface
  - `matplotlib.dates` - Date formatting on x-axes
  - `matplotlib.ticker.FuncFormatter` - Custom axis formatting
- **Version:** 3.5.0 or higher
- **Installation:** `pip install matplotlib`

**3. pandas** (`import pandas as pd`)
- **Purpose:** Data manipulation and time-series handling
- **Used for:** DataFrame operations, date indexing, data cleaning
- **Version:** 1.4.0 or higher
- **Installation:** `pip install pandas`

**4. numpy** (`import numpy as np`)
- **Purpose:** Numerical computations and array operations
- **Used for:** Mathematical calculations, statistical functions
- **Version:** 1.21.0 or higher
- **Installation:** `pip install numpy`

**5. statsmodels** (multiple imports)
- **Purpose:** Statistical modeling and time-series analysis
- **Used for:** ARIMA, SARIMA, Exponential Smoothing models
- **Submodules used:**
  - `statsmodels.tsa.arima.model.ARIMA` - ARIMA modeling
  - `statsmodels.tsa.statespace.sarimax.SARIMAX` - SARIMA modeling
  - `statsmodels.tsa.holtwinters` - Exponential smoothing (SES, Holt, Holt-Winters)
  - `statsmodels.tsa.stattools.adfuller` - Stationarity testing
  - `statsmodels.graphics.tsaplots` - ACF/PACF plotting
- **Version:** 0.13.0 or higher
- **Installation:** `pip install statsmodels`

**6. scikit-learn** (`from sklearn.metrics import ...`)
- **Purpose:** Model evaluation metrics
- **Used for:** Calculating RMSE, MAE for model performance
- **Functions used:**
  - `mean_squared_error` - Calculate MSE/RMSE
  - `mean_absolute_error` - Calculate MAE
- **Version:** 1.0.0 or higher
- **Installation:** `pip install scikit-learn`

**7. sys and warnings** (Built-in Python modules)
- **sys:** Command-line argument parsing
- **warnings:** Suppress non-critical warnings from models
- **No installation needed** - Part of Python standard library

### System Requirements

**Minimum:**
- CPU: Any modern processor (1 GHz+)
- RAM: 2 GB
- Storage: 100 MB free space
- Internet: Required for downloading stock data

**Recommended:**
- CPU: Multi-core processor (for faster model fitting)
- RAM: 4 GB or more
- Storage: 500 MB (for caching multiple stocks)
- Internet: Stable connection (1 Mbps+)

### Verifying Installation

Create a test script `check_dependencies.py`:
```python
def check_dependencies():
    dependencies = {
        'yfinance': 'yfinance',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'statsmodels': 'statsmodels',
        'scikit-learn': 'sklearn'
    }
    
    print("Checking dependencies...\n")
    all_installed = True
    
    for name, import_name in dependencies.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15s} - Version {version}")
        except ImportError:
            print(f"✗ {name:15s} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✓ All dependencies installed successfully!")
    else:
        print("\n✗ Some dependencies missing. Run: pip install -r requirements.txt")
    
    return all_installed

if __name__ == "__main__":
    check_dependencies()
```

Run it with:
```bash
python check_dependencies.py
```

Expected output:
```
Checking dependencies...

✓ yfinance        - Version 0.2.38
✓ matplotlib      - Version 3.8.2
✓ pandas          - Version 2.1.4
✓ numpy           - Version 1.26.3
✓ statsmodels     - Version 0.14.1
✓ scikit-learn    - Version 1.3.2

✓ All dependencies installed successfully!
```

### Troubleshooting Common Installation Issues

**Issue 1: "No module named 'yfinance'"**
```bash
# Solution:
pip install --upgrade yfinance
```

**Issue 2: Matplotlib won't display charts**
```bash
# On Linux, you might need:
sudo apt-get install python3-tk

# On macOS:
brew install python-tk
```

**Issue 3: statsmodels installation fails**
```bash
# Try upgrading pip first:
pip install --upgrade pip setuptools wheel

# Then install statsmodels:
pip install statsmodels
```

**Issue 4: Permission errors**
```bash
# Use --user flag:
pip install --user yfinance matplotlib pandas numpy statsmodels scikit-learn
```

**Issue 5: Conflicting versions**
```bash
# Create a virtual environment:
python -m venv time_series_env
source time_series_env/bin/activate  # On Windows: time_series_env\Scripts\activate
pip install yfinance matplotlib pandas numpy statsmodels scikit-learn
```

### Optional Enhancements

**For faster computations:**
```bash
pip install numba  # JIT compilation for numpy
```

**For Jupyter Notebook integration:**
```bash
pip install jupyter notebook ipython
```

**For enhanced data export:**
```bash
pip install openpyxl xlsxwriter  # Excel file support
```

### Project Structure

```
time-series-analysis/
│
├── Time-Series.py           # Main program
├── requirements.txt         # Dependency list
├── check_dependencies.py    # Verification script
├── README.md               # Documentation
│
├── data/                   # Downloaded CSV files
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   └── ...
│
└── output/                 # Generated charts (optional)
    ├── AAPL_technical.png
    ├── AAPL_forecast.png
    └── ...
```

### Quick Start After Installation

1. **Verify dependencies:**
```bash
python check_dependencies.py
```

2. **Test the program:**
```bash
python Time-Series.py AAPL 2023-01-01
```

3. **Run with forecasting:**
```bash
python Time-Series.py AAPL 2020-01-01 --smooth
```

---

## Part 1: Data Collection & Basic Analysis

### 1. **Stock Data Download** (`download_stock_data`)
**What it does:**
- Downloads historical stock price data from Yahoo Finance using the `yfinance` library
- Gets OHLCV data: Open, High, Low, Close, Volume
- Saves data to CSV file for later use
- Handles errors gracefully (invalid tickers, network issues)

**Example:**
```python
data = download_stock_data("AAPL", "2020-01-01", "2024-12-31")
# Downloads 5 years of Apple stock data
```

---

## Part 2: Technical Indicators & Analysis

### 2. **Moving Averages** (`calculate_moving_averages`)
**What it does:**
- Calculates Simple Moving Averages (SMA) over different time windows
- Default: 20-day (short-term), 50-day (medium-term), 200-day (long-term)
- Smooths out price fluctuations to identify trends

**Why it matters:**
- MA20: Shows recent price momentum
- MA50: Shows medium-term trend
- MA200: Shows long-term trend
- When short MA crosses above long MA = bullish signal (Golden Cross)
- When short MA crosses below long MA = bearish signal (Death Cross)

### 3. **Returns Analysis** (`calculate_returns`)
**What it does:**
- Calculates daily percentage changes in stock price
- Computes returns over 1, 7, and 30-day periods
- Shows profit/loss if you held the stock

**Example Output:**
- Daily Return: +2.5% means stock went up 2.5% today
- 30-day Return: +15% means stock is up 15% from 30 days ago

### 4. **Volatility** (`calculate_volatility`)
**What it does:**
- Measures how much the stock price fluctuates
- Uses standard deviation of returns
- Annualized to show yearly volatility
- Rolling 30-day window shows changing volatility over time

**Why it matters:**
- High volatility = risky stock, big price swings
- Low volatility = stable stock, small price movements
- Volatility typically increases during market crashes

### 5. **RSI - Relative Strength Index** (`calculate_rsi`)
**What it does:**
- Momentum oscillator measuring speed and magnitude of price changes
- Scale: 0-100
- Compares average gains vs average losses over 14 days

**Interpretation:**
- RSI > 70: Overbought (stock might be too expensive, could drop)
- RSI < 30: Oversold (stock might be too cheap, could rise)
- RSI = 50: Neutral

### 6. **Bollinger Bands** (`calculate_bollinger_bands`)
**What it does:**
- Creates a channel around the moving average
- Upper band = MA + (2 × standard deviation)
- Lower band = MA - (2 × standard deviation)
- Shows if price is high or low relative to recent history

**Trading Signals:**
- Price near upper band = potentially overbought
- Price near lower band = potentially oversold
- Bands squeeze together = low volatility, big move coming
- Bands expand = high volatility, trend in progress

### 7. **Trend Detection** (`detect_trend`)
**What it does:**
- Uses 50-day and 200-day moving average crossover
- Signal = +1 (bullish), -1 (bearish), 0 (neutral)

**Golden Cross:** 50-day MA crosses above 200-day MA → Strong buy signal
**Death Cross:** 50-day MA crosses below 200-day MA → Strong sell signal

---

## Part 3: Visualization

### 8. **Basic Stock Charts** (`plot_stock_data`)
**Creates:**
- Top chart: Closing price over time with date formatting
- Bottom chart: Trading volume (how many shares traded)

**Why volume matters:**
- High volume + price increase = strong buying pressure
- High volume + price decrease = strong selling pressure
- Low volume moves are less reliable

### 9. **Technical Analysis Charts** (`plot_timeseries_analysis`)
**Creates three subplots:**
1. **Price chart** with MA20, MA50, MA200, and Bollinger Bands
2. **RSI indicator** with overbought/oversold zones
3. **Volume bars**

This gives you a complete technical view at a glance.

---

## Part 4: Statistical Tests

### 10. **Stationarity Test** (`check_stationarity`)
**What it does:**
- Performs Augmented Dickey-Fuller (ADF) test
- Tests if the time series has a stable mean and variance

**Why it matters:**
- ARIMA models require stationary data
- Stock prices are usually non-stationary (they trend)
- Returns are usually stationary (they fluctuate around zero)
- If non-stationary, we need to "difference" the data

**Results:**
- p-value < 0.05 = stationary ✓
- p-value ≥ 0.05 = non-stationary, needs differencing

### 11. **ACF & PACF Plots** (`plot_acf_pacf`)
**What they show:**
- **ACF (Autocorrelation):** How today's price relates to past prices
- **PACF (Partial Autocorrelation):** Direct relationship, removing intermediate effects

**Why it matters:**
- Helps select ARIMA parameters (p, d, q)
- ACF shows moving average (MA) order
- PACF shows autoregressive (AR) order

---

## Part 5: Forecasting Models

## ARIMA Family Models

### 12. **ARIMA Model** (`fit_arima_model`)
**What it is:**
- **AR**uto**R**egressive **I**ntegrated **M**oving **A**verage
- Three parameters: (p, d, q)
  - **p:** Number of lag observations (how many past values to use)
  - **d:** Degree of differencing (how many times to difference data)
  - **q:** Size of moving average window

**How it works:**
1. Differences the data to make it stationary
2. Uses past values to predict future values
3. Applies moving average to smooth errors

**Example:**
- ARIMA(1,1,1): Uses 1 lag, differences once, 1 MA term
- Good for data with trends but no seasonality

### 13. **SARIMA Model** (`fit_sarima_model`)
**What it is:**
- Seasonal ARIMA - handles repeating patterns
- Parameters: (p,d,q) × (P,D,Q,s)
  - (p,d,q): Same as ARIMA
  - (P,D,Q,s): Seasonal components
  - **s:** Season length (12 for monthly, 252 for daily yearly data)

**When to use:**
- Stock shows seasonal patterns (e.g., "Santa Rally" in December)
- Monthly/quarterly patterns
- Day-of-week effects

### 14. **Auto ARIMA** (`auto_arima_selection`)
**What it does:**
- Tests hundreds of ARIMA parameter combinations
- Selects best model based on AIC (Akaike Information Criterion)
- Lower AIC = better model

**Grid search:**
- Tries p from 0 to max_p
- Tries d from 0 to max_d
- Tries q from 0 to max_q
- Finds optimal (p,d,q) automatically

---

## Exponential Smoothing Models

### 15. **Simple Exponential Smoothing (SES)** (`fit_ses_model`)
**What it is:**
- Weighted average giving more weight to recent observations
- Single parameter: α (alpha) = smoothing level (0-1)
- Higher α = more weight to recent data

**Best for:**
- Data with no trend or seasonality
- Short-term forecasting
- Rapidly changing markets

**Formula:**
```
Forecast = α × (current value) + (1-α) × (previous forecast)
```

### 16. **Holt's Linear Trend Model** (`fit_holt_model`)
**What it is:**
- Extends SES to handle trends
- Two parameters:
  - **α (alpha):** Level smoothing
  - **β (beta):** Trend smoothing

**Best for:**
- Data with linear trends
- No seasonality
- Stocks in clear uptrend or downtrend

**Damped Trend Option:**
- Flattens the trend for long-term forecasts
- More conservative predictions
- Prevents unrealistic exponential forecasts

### 17. **Holt-Winters Model** (`fit_holtwinters_model`)
**What it is:**
- Triple Exponential Smoothing
- Handles level, trend, AND seasonality
- Three parameters: α, β, γ (gamma for seasonality)

**Configuration options:**
- **Trend:** 'add' (linear), 'mul' (exponential), or None
- **Seasonal:** 'add' (constant amplitude), 'mul' (proportional amplitude), or None
- **Periods:** Length of seasonal cycle (21 for monthly, 252 for yearly)

**Best for:**
- Complex time series
- Clear seasonal patterns
- Most comprehensive exponential smoothing approach

### 18. **Auto Holt-Winters** (`auto_holtwinters_selection`)
**What it does:**
- Tests all combinations of:
  - Trend types: additive, multiplicative, none
  - Seasonal types: additive, multiplicative, none
  - Seasonal periods: 7, 12, 21, 30, 60, 90, 126, 252 days
- Selects best configuration based on AIC

---

## Part 6: Model Evaluation & Comparison

### 19. **Performance Metrics**
All models calculate four key metrics:

**RMSE (Root Mean Square Error):**
- Average prediction error in dollars
- Penalizes large errors heavily
- Lower is better
- Example: RMSE = $5 means predictions off by ~$5 on average

**MAE (Mean Absolute Error):**
- Average absolute prediction error
- Treats all errors equally
- More interpretable than RMSE
- Example: MAE = $3 means average error is $3

**MAPE (Mean Absolute Percentage Error):**
- Average error as percentage
- Good for comparing across different stocks
- Example: MAPE = 2% means predictions off by 2% on average

**AIC (Akaike Information Criterion):**
- Balances model accuracy vs complexity
- Penalizes models with too many parameters
- Use for comparing models (lower is better)
- Helps prevent overfitting

### 20. **Train/Test Split**
**How it works:**
- 80% of data = Training set (to fit the model)
- 20% of data = Test set (to evaluate predictions)
- Models never see test data during training

**Why it matters:**
- Prevents overfitting
- Shows real-world performance
- If train performance >> test performance = overfitting

### 21. **Model Comparison** (`compare_all_models`)
**What it does:**
- Creates a comparison table of all fitted models
- Shows RMSE, MAE, MAPE, AIC for each
- Declares a "winner" for each metric
- Provides performance summary

**Example output:**
```
Metric      ARIMA(1,1,1)    SES             Holt            Winner
---------------------------------------------------------------
RMSE        12.45           15.23           11.87           Holt
MAE         9.34            11.56           9.12            Holt
MAPE        2.15            2.67            2.01            Holt
AIC         5234.12         5456.78         5198.45         Holt
```

---

## Part 7: Forecasting

### 22. **How Forecasting Works**

**Step 1: Historical Analysis**
- Model learns patterns from training data (80%)
- Identifies trends, seasonality, autocorrelation

**Step 2: Validation**
- Tests predictions on test data (20%)
- Calculates error metrics
- Adjusts if necessary

**Step 3: Future Forecasting**
- Uses fitted model to predict next 30 days
- Generates point forecasts
- Uncertainty increases with time horizon

### 23. **Forecast Visualization**
**Charts show:**
- **Blue line:** Training data (what model learned from)
- **Green line:** Actual test data (ground truth)
- **Red dashed line:** Model predictions on test set
- **Orange dotted line:** Future forecast (next 30 days)

**Interpreting the charts:**
- If red matches green closely = good model
- If red diverges from green = poor predictions
- Orange forecast shows expected price movement

---

## Part 8: Usage & Workflow

### 24. **Command-Line Interface**

**Basic usage:**
```bash
python Time-Series.py <TICKER> [START_DATE] [END_DATE] [OPTIONS]
```

**Examples:**

**1. Just technical analysis (no forecasting):**
```bash
python Time-Series.py AAPL
```
Shows charts and indicators, no predictions.

**2. ARIMA forecasting:**
```bash
python Time-Series.py AAPL 2020-01-01 --arima
```
Downloads data from 2020, fits ARIMA(1,1,1), forecasts 30 days.

**3. Auto-select best ARIMA:**
```bash
python Time-Series.py TSLA 2020-01-01 --auto
```
Tests hundreds of ARIMA models, picks best one.

**4. All exponential smoothing models:**
```bash
python Time-Series.py GOOGL 2020-01-01 --smooth
```
Fits SES, Holt, and Holt-Winters, compares them.

**5. Auto-optimize Holt-Winters:**
```bash
python Time-Series.py MSFT 2020-01-01 --auto-smooth
```
Finds best seasonal configuration automatically.

**6. Compare everything:**
```bash
python Time-Series.py NVDA 2020-01-01 --arima --sarima --smooth
```
Fits 5 models (ARIMA, SARIMA, SES, Holt, Holt-Winters), compares all.

### 25. **Complete Workflow Example**

**Input:**
```bash
python Time-Series.py AAPL 2020-01-01 2024-12-31 --arima --smooth
```

**What happens:**

1. **Download data** → 5 years of Apple stock (2020-2024)

2. **Calculate indicators:**
   - Moving averages (20, 50, 200-day)
   - Daily returns
   - 30-day volatility
   - RSI
   - Bollinger Bands
   - Trend signals

3. **Display charts:**
   - Price + Volume chart
   - Technical analysis chart (MAs, RSI, Bollinger Bands)

4. **Print summary:**
   - Average daily return: 0.12%
   - Total return: 245%
   - Current volatility: 28%
   - RSI: 65 (Neutral)
   - Trend: Bullish (Golden Cross)

5. **Test stationarity:**
   - Raw prices: Non-stationary (p-value = 0.87)
   - Differenced prices: Stationary (p-value = 0.001)

6. **Show ACF/PACF plots** → Help identify patterns

7. **Fit ARIMA(1,1,1):**
   - Training: 2020-2023 (80%)
   - Testing: 2023-2024 (20%)
   - RMSE: $12.45
   - MAPE: 2.15%
   - Forecast next 30 days

8. **Fit SES:**
   - Alpha: 0.85 (optimized)
   - RMSE: $15.23
   - MAPE: 2.67%

9. **Fit Holt:**
   - Alpha: 0.82, Beta: 0.15
   - RMSE: $11.87
   - MAPE: 2.01%

10. **Fit Holt-Winters:**
    - Alpha: 0.78, Beta: 0.12, Gamma: 0.05
    - Seasonal period: 21 days
    - RMSE: $10.34
    - MAPE: 1.87%

11. **Compare models:**
    - Winner: Holt-Winters (lowest RMSE, MAE, MAPE)
    - All models plotted together

12. **Output:** CSV file with all data saved

---

## Part 9: Interpreting Results

### 26. **Making Investment Decisions**

**⚠️ IMPORTANT DISCLAIMER:**
This is an educational tool, NOT investment advice. Always:
- Do your own research
- Consider multiple factors (company fundamentals, news, economic conditions)
- Diversify your portfolio
- Never invest money you can't afford to lose
- Past performance doesn't guarantee future results

**How to use the analysis:**

**Technical Indicators:**
- Multiple bullish signals = potential buy opportunity
- Multiple bearish signals = potential sell/avoid
- Conflicting signals = wait for clarity

**Forecast Models:**
- If multiple models agree on direction = stronger signal
- Large forecast uncertainty = risky prediction
- Consider ensemble (average) of multiple models

**Example scenario:**
```
✓ RSI: 35 (Oversold)
✓ Price near lower Bollinger Band
✓ Golden Cross forming (MA50 crossing above MA200)
✓ All models forecast +15% in 30 days
✓ Low volatility (stable)
→ Potential buy signal (but verify fundamentals!)

vs.

✗ RSI: 75 (Overbought)
✗ Price at upper Bollinger Band
✗ Death Cross forming
✗ Models forecast -10% in 30 days
✗ High volatility (risky)
→ Avoid or consider selling
```

### 27. **Model Strengths & Weaknesses**

**ARIMA:**
- ✓ Good for short-term forecasts
- ✓ Captures autocorrelation well
- ✓ Widely used and understood
- ✗ Struggles with sudden changes
- ✗ Assumes linear relationships
- ✗ No seasonality handling

**SARIMA:**
- ✓ Handles seasonal patterns
- ✓ Better for cyclical stocks
- ✓ More flexible than ARIMA
- ✗ Requires more data
- ✗ Computationally intensive
- ✗ Can overfit

**SES:**
- ✓ Simple and fast
- ✓ Good for stable series
- ✓ Few parameters to tune
- ✗ No trend or seasonality
- ✗ Only for very short-term
- ✗ Naive approach

**Holt:**
- ✓ Handles trends well
- ✓ Fast computation
- ✓ Damped option prevents unrealistic forecasts
- ✗ No seasonality
- ✗ Assumes linear trends
- ✗ Sensitive to outliers

**Holt-Winters:**
- ✓ Most comprehensive
- ✓ Handles trend + seasonality
- ✓ Flexible configurations
- ✗ Needs longer data history
- ✗ More parameters to optimize
- ✗ Can be unstable

---

## Part 10: Advanced Tips

### 28. **Choosing the Right Model**

**For stocks with clear trends:**
→ Use Holt or ARIMA

**For cyclical/seasonal stocks:**
→ Use SARIMA or Holt-Winters

**For volatile tech stocks:**
→ Use shorter time windows, ensemble methods

**For stable blue-chip stocks:**
→ Any model should work, prefer simpler ones

### 29. **Improving Predictions**

**1. More data:**
- Minimum: 2 years for ARIMA
- Recommended: 3-5 years
- Seasonal models: Need 2+ complete cycles

**2. Feature engineering:**
- Add external factors (market indices, sector performance)
- Include macroeconomic indicators (interest rates, GDP)
- Consider sentiment analysis from news

**3. Ensemble methods:**
- Average predictions from multiple models
- Weight by past performance
- Reduces individual model bias

**4. Regular retraining:**
- Markets change over time
- Retrain models monthly or quarterly
- Adjust for structural breaks (COVID, policy changes)

### 30. **Common Pitfalls**

**❌ Overfitting:**
- Using too many parameters
- Solution: Use AIC/BIC for model selection

**❌ Ignoring non-stationarity:**
- Not differencing when needed
- Solution: Always test stationarity first

**❌ Small sample size:**
- Not enough data for complex models
- Solution: Use simpler models or get more data

**❌ Look-ahead bias:**
- Using future information in training
- Solution: Strict train/test split, no data leakage

**❌ Ignoring outliers:**
- Market crashes, stock splits distort models
- Solution: Clean data, handle special events

---

## Summary

This program is a complete stock analysis and forecasting toolkit that:

1. **Downloads** historical stock data
2. **Analyzes** with technical indicators (RSI, MAs, Bollinger Bands, etc.)
3. **Visualizes** trends and patterns with professional charts
4. **Tests** for statistical properties (stationarity, autocorrelation)
5. **Forecasts** future prices using 5 different models
6. **Compares** model performance with robust metrics
7. **Outputs** actionable insights and predictions

It combines traditional technical analysis with modern time-series forecasting to give you a comprehensive view of stock behavior and potential future movement.
