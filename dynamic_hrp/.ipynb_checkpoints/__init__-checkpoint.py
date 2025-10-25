"""
dynamic_hrp — Regime-aware Dynamic HRP backtesting toolkit.

Submodules:
- io_data:      load raw CSVs and build cleaned price panels
- standardize:  robust date parsing, ffill pruning, weekly resampling, bond-quote parsing
- universe:     select investable universe & feature set from the wide panel
- returns:      compute daily/weekly log returns and apply execution delay
- signals:      build TSMOM signals and optional smoothing
- features:     build rolling regime features and expanding standardized versions
- hmm_wf:       walk-forward HMM training & inference (no look-ahead)
- hrp:          HRP allocation variants (variance, CVaR) with clustering utilities
- backtests:    dynamic HRP, equal-weight, static HRP baselines
"""
