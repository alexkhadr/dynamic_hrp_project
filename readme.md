# Dynamic HRP Project — Regime-Aware Portfolio Allocation

This project implements a Dynamic Hierarchical Risk Parity (HRP) strategy that adapts portfolio weights based on Hidden Markov Model (HMM)-detected market regimes. It combines data integration, feature engineering, regime classification, and dynamic portfolio optimization to evaluate performance against static and equal-weight benchmarks.

### PROJECT PURPOSE

The goal of this project is to build and test a regime-switching, multi-asset portfolio that dynamically adjusts its composition according to macro-financial conditions. <br>

By detecting shifts between Trending, Neutral, and Crisis market regimes, the model aims to:
- Reduce drawdowns during crises
- Allocate more efficiently during stable or trending markets
- Improve risk-adjusted returns compared to static HRP and equal-weight portfolios

### DATA DESCRIPTION

The project integrates several Bloomberg-style CSV datasets located in the Data folder. <br>

Dataset: AT-03_Energy_Metals_Comdty_Future_Daily_2000_2025.csv <br>
Description: Energy and metals futures <br>
Example tickers: CL1, NG1, GC1 <br>

Dataset: AT-04_Daily_Spot_Prices_G10_FX_Pairs_Daily_2000_2025.csv <br>
Description: G10 spot FX pairs <br>
Example tickers: EURUSD, USDJPY, GBPUSD <br>

Dataset: AT-15_NOB_Spread_Compenents_Daily_2005_2025.csv <br>
Description: 10-Year U.S. Treasury future <br>
Example ticker: TY1 <br>

Dataset: AT-16_Euro_Bond_Future_Daily_2005_2025.csv <br>
Description: Euro-Bund future <br>
Example ticker: RX1 <br>

Dataset: AT-39_Equity_Futures_Daily_1990_2025.csv <br>
Description: Global equity index futures <br>
Example tickers: ES1 (S&P 500), VG1 (Euro Stoxx 50), NK1 (Nikkei) <br>

Dataset: AT-46_Volatility_Index_Daily_1990_2025.csv <br>
Description: Volatility indices <br>
Example tickers: VIX, V2X <br>

Each dataset is read, cleaned, and standardized into a single unified daily and weekly price panel.

### REPOSITORY STRUCTURE

dynamic_hrp_project/ <br>
│<br>
├── dynamic_hrp/ # Core Python package<br>
│ ├── init.py<br>
│ ├── io_data.py # Load and clean raw CSVs into unified panels<br>
│ ├── standardize.py # Helper functions for date parsing and cleaning<br>
│ ├── universe.py # Select investable assets and feature panels<br>
│ ├── returns.py # Compute daily and weekly log returns<br>
│ ├── signals.py # Generate trend-following (TSMOM) signals<br>
│ ├── features.py # Build and standardize HMM regime features<br>
│ ├── hmm_wf.py # Hidden Markov Model (walk-forward inference)<br>
│ ├── hrp.py # HRP allocation models (variance and CVaR)<br>
│ ├── backtests.py # Backtest logic for dynamic HRP and benchmarks<br>
│ └── perf.py # Performance metrics and statistics<br>
│<br>
├── Data/ <br>
│ ├── AT-03_Energy_Metals_Comdty_Future_Daily_2000_2025.csv<br>
│ ├── AT-04_Daily_Spot_Prices_G10_FX_Pairs_Daily_2000_2025.csv<br>
│ ├── AT-15_NOB_Spread_Compenents_Daily_2005_2025.csv<br>
│ ├── AT-16_Euro_Bond_Future_Daily_2005_2025.csv<br>
│ ├── AT-39_Equity_Futures_Daily_1990_2025.csv<br>
│ └── AT-46_Volatility_Index_Daily_1990_2025.csv<br>
├── Figures/ <br>
│ ├── cumulative_pnl.png <br>
│ ├── hist_by_regime.png<br>
│ └── hist_by_strategy_and_regime.png<br>
│<br>
├── main.py # Main execution file<br>
└── README.md # Project documentation<br>


### HOW TO RUN THE PROJECT

Navigate to the project folder. <br>
Example: <br>
cd "C:/Users/alexk/Desktop/University of Toronto/Dynamic Data Science/dynamic_hrp_project" <br>

Install required libraries. <br>
python -m pip install -r requirements.txt <br>
OR <br>
pip install pandas numpy matplotlib scipy hmmlearn scikit-learn <br>

Run the project. <br>
python main.py <br>

### WHAT THE SCRIPT DOES

When you run main.py, it will: <br>

- Load all raw CSVs from the Data folder.
- Build unified price panels for FX, equities, rates, commodities, and volatility.
- Generate weekly trend-following (TSMOM) signals.
- Construct HMM regime features such as volatility, correlation, dispersion, and skewness.
- Train a Hidden Markov Model to identify regimes (Trending, Neutral, Crisis).
- Backtest Dynamic HRP portfolios that switch allocation rules depending on the regime.
- Compare performance against static HRP and equal-weight portfolios.
- Output plots and statistics such as cumulative returns, Sharpe ratio, CAGR, and drawdown. Graphs are saved in the Figures folder and tables are saved in the Tables folder

### EXAMPLE OUTPUTS

Regime labels: <br>
Trending, Neutral, Crisis <br>

Performance statistics: <br>
Mean weekly return: 0.0018 <br>
Volatility: 0.0123 <br>
Sharpe ratio: 0.73 <br> 
CAGR: 6.5% <br>
Maximum drawdown: 9.8% <br>

Outputs include cumulative P&L plots comparing: 

- Dynamic HRP
- Static HRP (Variance)
- Equal Weight portfolio

### INTERPRETATION

Trending Regime: Low volatility and low correlation — model allocates more to risk assets. <br>
Neutral Regime: Moderate volatility — balanced allocation. <br>
Crisis Regime: High volatility and high correlation — defensive allocation using CVaR HRP. <br>

This regime-adaptive approach aims to maintain exposure in favorable environments while mitigating losses during crises.

### REQUIREMENTS

- Python 3.9
- pandas
- numpy
- scipy
- scikit-learn
- hmmlearn
- matplotlib


### FUTURE IMPROVEMENTS

- Integrate macroeconomic or sentiment data into the HMM
- Add volatility-targeting for smoother portfolio risk
- Include transaction cost modeling
- Extend to multi-frequency (daily, weekly, monthly) adaptive frameworks
