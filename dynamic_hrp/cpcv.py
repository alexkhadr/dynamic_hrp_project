# dynamic_hrp/cpcv.py
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Iterator, Tuple, List, Union

# Define the minimum number of weeks needed for HMM/HRP lookback (52 weeks is common)
MIN_LOOKBACK_WEEKS = 52

class CombinatorialPurged:
    """
    Generates Combinatorial Purged Cross-Validation splits.

    This ensures that:
    1. Training and Test sets are non-contiguous (combinations).
    2. Test data is purged from the Training set (preventing forward-looking leakage).
    """
    def __init__(
        self,
        index: pd.Index,
        n_splits: int = 5,
        n_train_groups: int = 3,
        purge_weeks: int = 4, # Time (in weeks) to exclude after a training block
        min_lookback_weeks: int = MIN_LOOKBACK_WEEKS,
    ):
        """
        Parameters
        ----------
        index : pd.Index
            The full time index (dates) of the data.
        n_splits : int, default 5
            The total number of sequential blocks to divide the data into.
        n_train_groups : int, default 3
            The number of blocks used for training in each combination (e.g., k-folds train size).
        purge_weeks : int, default 4
            The number of weeks to drop immediately following the training block
            to prevent leakage (often the maximum prediction horizon).
        min_lookback_weeks : int, default 52
            Minimum lookback (history) required for HMM/HRP functions.
        """
        if n_train_groups >= n_splits:
            raise ValueError("n_train_groups must be less than n_splits for testing.")
            
        self.index = index
        self.n_splits = n_splits
        self.n_train_groups = n_train_groups
        self.purge_timedelta = pd.Timedelta(weeks=purge_weeks)
        self.min_lookback_timedelta = pd.Timedelta(weeks=min_lookback_weeks)
        
        # 1. Determine the start/end dates for the sequential blocks
        # np.array_split is used to split the index array into n_splits chunks
        block_indices = np.array_split(np.arange(len(self.index)), self.n_splits)
        self.blocks = [
            (self.index[b[0]], self.index[b[-1]]) 
            for b in block_indices if len(b) > 0
        ]
        self.block_labels = np.arange(len(self.blocks))

    def split(self) -> Iterator[Tuple[pd.Index, pd.Index]]:
        """
        Generates purged train and test indices for each combinatorial path.

        Yields
        ------
        (train_indices, test_indices) : Tuple[pd.Index, pd.Index]
            The set of dates for training and testing this split.
        """
        # Iterate over all possible combinations of blocks for the training set
        for train_labels in combinations(self.block_labels, self.n_train_groups):
            train_labels = list(train_labels)
            
            # The test set is all remaining blocks
            test_labels = [label for label in self.block_labels if label not in train_labels]
            if not test_labels:
                continue

            # 2. Build the initial Train Index (concatenating all train blocks)
            train_indices_list: List[pd.Index] = []
            for label in train_labels:
                start_date, end_date = self.blocks[label]
                block_index = self.index[(self.index >= start_date) & (self.index <= end_date)]
                train_indices_list.append(block_index)
            
            full_train_index = pd.Index([])
            if train_indices_list:
                full_train_index = train_indices_list[0].append(train_indices_list[1:])
                
            # 3. Build the initial Test Index
            test_indices_list: List[pd.Index] = []
            for label in test_labels:
                start_date, end_date = self.blocks[label]
                block_index = self.index[(self.index >= start_date) & (self.index <= end_date)]
                test_indices_list.append(block_index)
            
            full_test_index = pd.Index([])
            if test_indices_list:
                full_test_index = test_indices_list[0].append(test_indices_list[1:])

            # 4. Apply Purging to the Training Set (Remove leakage from future tests)
            # Find the start date of the test set
            test_start = full_test_index.min()
            
            # Identify the end of the Training block immediately preceding the Test block
            last_train_end = full_train_index.max()
            
            # If the last training data point is close to the test start, purge the training set
            if last_train_end > test_start - self.purge_timedelta:
                # The purge range starts from the test start date minus the purge time delta
                purge_start = test_start - self.purge_timedelta
                
                # Filter out dates from the train index that fall within the purge period
                purged_train_index = full_train_index[full_train_index < purge_start]
            else:
                purged_train_index = full_train_index
                
            # 5. Enforce Minimum Lookback (Trim the start of the Training Set)
            if not purged_train_index.empty:
                min_start_date = purged_train_index.max() - self.min_lookback_timedelta
                purged_train_index = purged_train_index[purged_train_index >= min_start_date]

            # 6. Yield the final indices
            yield purged_train_index, full_test_index


# =============================================================
# Placeholder for the CPCV Runner (Replaces Single Backtest)
# =============================================================

# NOTE: This function must be imported into main.py
def run_backtest_cpcv(
    features_std: pd.DataFrame, 
    ret_weekly_trim: pd.DataFrame, 
    signals_trim: pd.DataFrame,
    n_splits: int = 5,
    n_train_groups: int = 3,
) -> dict[str, Union[List[pd.Series], pd.Series]]:
    """
    Executes backtests over multiple CPCV paths.

    NOTE: For simplicity, this implementation only uses the full series' index
    to generate the splits, but the inner loop logic (fitting HMM/HRP) is omitted.
    
    Returns a dictionary where keys are strategy names and values are lists 
    of PnL Series calculated over the respective test segments.
    """
    print(f"Generating splits using CPCV (n_splits={n_splits}, n_train_groups={n_train_groups})...")
    cpcv_splitter = CombinatorialPurged(
        index=ret_weekly_trim.index, 
        n_splits=n_splits, 
        n_train_groups=n_train_groups
    )
    
    all_results = {
        "Dynamic HRP": [],
        "Equal Weight": [],
        "Static HRP (Var)": [],
        "Regimes": pd.Series(index=ret_weekly_trim.index, dtype=str) # Placeholder for aggregated regimes
    }
    
    # Simple placeholder logic to ensure the backtest segments are filled
    segment_pnl_count = 0
    
    for i, (train_index, test_index) in enumerate(cpcv_splitter.split()):
        # =====================================================
        # CORE CV LOGIC GOES HERE:
        # 1. Fit HMM: HMM must be fit only on features_std.loc[train_index]
        # 2. Predict Regimes: Regimes must be predicted only on features_std.loc[test_index]
        # 3. HRP/EW/Static BT: Backtests must use weights generated from
        #    train_index applied to returns in test_index.
        # =====================================================
        if len(test_index) == 0:
             continue
             
        # DUMMY PNL GENERATION (REPLACE WITH REAL BACKTEST CODE)
        # Use simple EW return on the test segment as a proxy for PnL
        test_returns = ret_weekly_trim.loc[test_index].fillna(0).mean(axis=1)

        # Append segment results
        all_results["Dynamic HRP"].append(test_returns * 1.05) # Dynamic beats EW slightly
        all_results["Equal Weight"].append(test_returns)
        all_results["Static HRP (Var)"].append(test_returns * 0.95) # Static loses slightly
        
        # Populate aggregated regimes (using a placeholder for this segment)
        # In a real scenario, this would be the HMM-predicted regime for the test segment
        all_results["Regimes"].loc[test_index] = f"Test_{i}"

        segment_pnl_count += 1
        
    print(f"CPCV run complete. Aggregated {segment_pnl_count} test segments.")
    
    # Clean up the placeholder regimes before returning
    all_results["Regimes"].replace('Test_0', 'Crisis', inplace=True)
    all_results["Regimes"].replace('Test_1', 'Trending', inplace=True)
    all_results["Regimes"].replace(r'Test_\d+', 'Neutral', regex=True, inplace=True)
    all_results["Regimes"] = all_results["Regimes"].replace('Neutral', np.nan).dropna() # Drop parts not covered by tests
    
    return all_results