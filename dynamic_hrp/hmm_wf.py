# -------------------------------------------------------------
# Hidden Markov Model (HMM) Walk-Forward Module
# -------------------------------------------------------------
# This module fits a Gaussian Hidden Markov Model (HMM) to regime features
# using a walk-forward approach. It provides smoothed state sequences,
# state labeling ("Trending", "Neutral", "Crisis"), and state probabilities.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# -------------------------------------------------------------
# Helper: enforce minimum regime duration
# -------------------------------------------------------------
def _enforce_min_duration(path: np.ndarray, min_len: int = 2) -> np.ndarray:
    """
    Post-process a discrete state sequence so that any regime run shorter than
    `min_len` is merged into the nearest longer neighbor. This prevents
    unrealistic one-week "blips" in regime labeling.
    """
    if min_len <= 1 or len(path) == 0:
        return path.copy()

    x = path.copy()
    n = len(x)

    # Identify contiguous runs of identical states
    runs = []
    start = 0
    for i in range(1, n):
        if x[i] != x[i - 1]:
            runs.append((x[start], start, i - 1))
            start = i
    runs.append((x[start], start, n - 1))  # include final run

    changed = True
    while changed:
        changed = False
        new_runs = []
        for idx, (state, a, b) in enumerate(runs):
            length = b - a + 1
            # If this run is shorter than min_len, merge it
            if length < min_len:
                left  = runs[idx - 1] if idx > 0 else None
                right = runs[idx + 1] if idx < len(runs) - 1 else None
                candidates = []
                # Collect left and right neighbor candidates with metadata
                if left is not None:
                    candidates.append(("L", left[0], left[1], left[2], left[2] - left[1] + 1, a - left[2]))
                if right is not None:
                    candidates.append(("R", right[0], right[1], right[2], right[2] - right[1] + 1, right[1] - b))
                if not candidates:
                    continue
                # Choose the neighbor with the longer run (ties broken by proximity)
                side, new_state, *_ = sorted(candidates, key=lambda t: (-t[4], t[5]))[0]
                # Replace the short run with the chosen neighbor's state
                x[a:b+1] = new_state
                changed = True
            else:
                new_runs.append((state, a, b))
        if changed:
            # Recompute contiguous runs after modification
            runs = []
            start = 0
            for i in range(1, n):
                if x[i] != x[i - 1]:
                    runs.append((x[start], start, i - 1))
                    start = i
            runs.append((x[start], start, n - 1))

    return x


# -------------------------------------------------------------
# Helper: label integer HMM states with interpretable names
# -------------------------------------------------------------
def label_states_by_stats(features: pd.DataFrame, states: pd.Series) -> dict[int, str]:
    """
    Assign intuitive labels to HMM states based on average feature values.

    Logic:
      - Compute mean volatility (vol_mean) and correlation (corr_mean_offdiag)
        for each HMM state.
      - Sort states by their average values.
      - Label them accordingly:
            lowest → "Trending"
            middle → "Neutral"
            highest → "Crisis"

    Returns
    -------
    mapping : dict[int, str]
        Maps integer HMM states to string labels.
    """
    df = features.copy()
    df["state"] = states.values

    # Compute per-state feature averages
    grp = df.groupby("state").mean(numeric_only=True)

    # Focus on key features if available
    cols = [c for c in ["vol_mean", "corr_mean_offdiag"] if c in grp.columns]
    comp = grp[cols].mean(axis=1) if cols else grp.mean(axis=1)

    # Sort states by "calmness" → high vol/corr = turbulent
    order = comp.sort_values().index.tolist()
    mapping: dict[int, str] = {}

    # Assign descriptive names depending on number of components
    if len(order) == 3:
        mapping[order[0]] = "Trending"
        mapping[order[1]] = "Neutral"
        mapping[order[2]] = "Crisis"
    elif len(order) == 2:
        mapping[order[0]] = "Calm"
        mapping[order[1]] = "Turbulent"
    else:
        mapping[order[0]] = "Regime"
    return mapping


# -------------------------------------------------------------
# Core: Walk-forward Hidden Markov Model fitting
# -------------------------------------------------------------
def fit_hmm_walkforward(
    features_std: pd.DataFrame,
    n_components: int = 3,
    covariance_type: str = "diag",
    refit_every_weeks: int = 4,
    min_train_weeks: int = 104,
    min_state_duration: int = 2,
    random_state: int = 42,
):
    """
    Perform walk-forward Hidden Markov Model fitting and labeling.

    Each iteration:
      • Fits a Gaussian HMM on all available data up to time t-1.
      • Decodes (Viterbi) the most likely state sequence.
      • Assigns the last inferred state to week t.
      • Optionally enforces a minimum regime duration.

    Parameters
    ----------
    features_std : pd.DataFrame
        Standardized regime features (no NaNs, standardized via expanding z-scores).
    n_components : int
        Number of hidden states.
    covariance_type : str
        Covariance type ('diag' for diagonal, 'full' for full covariance).
    refit_every_weeks : int
        How often to refit the model (rolling window frequency).
    min_train_weeks : int
        Minimum number of weeks required before the first fit.
    min_state_duration : int
        Minimum length (in weeks) for any regime (post-processed).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "state_int"         : raw integer state series
        "state_int_smooth"  : post-processed series with enforced duration
        "state_label"       : human-readable state labels
        "label_map"         : mapping from integer to label
        "state_proba"       : per-state posterior probabilities
    """
    # --- Clean and prepare data ---
    X = (
        features_std.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )
    dates = X.index
    if len(dates) <= min_train_weeks:
        raise ValueError(
            f"Not enough history: {len(dates)} <= min_train_weeks={min_train_weeks}"
        )

    # Preallocate storage for state sequence and probabilities
    state_int = pd.Series(index=dates, dtype="Int64")
    proba_df = pd.DataFrame(index=dates, columns=[f"p_state_{k}" for k in range(n_components)], dtype=float)

    # Initialize variables for walk-forward fitting
    first_fit_ix = min_train_weeks
    model = None
    last_refit = -10**9

    # --- Walk-forward loop over time ---
    for t_idx in range(first_fit_ix, len(dates)):
        # Refit model periodically or on first iteration
        if (t_idx - last_refit) >= refit_every_weeks or model is None:
            X_train = X.iloc[:t_idx].values
            model = GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=200,
                random_state=random_state,
                verbose=False,
            )
            model.fit(X_train)
            last_refit = t_idx

        # Decode up to current point and assign current state
        X_up_to_t = X.iloc[: t_idx + 1].values
        _, vt_path = model.decode(X_up_to_t, algorithm="viterbi")
        state_int.iloc[t_idx] = int(vt_path[-1])

        # Try to extract posterior probabilities (state confidence)
        try:
            post = model.predict_proba(X_up_to_t)
            proba_df.iloc[t_idx, :] = post[-1, :]
        except Exception:
            # If model.predict_proba fails, use lower-level internals
            try:
                framelogprob = model._compute_log_likelihood(X_up_to_t)
                posteriors = model._compute_posteriors(framelogprob)[0]
                proba_df.iloc[t_idx, :] = posteriors[-1, :]
            except Exception:
                pass

    # --- Fill gaps and enforce minimum regime duration ---
    state_int_filled = state_int.bfill().ffill().fillna(0).astype(int)
    x_vals = state_int_filled.values

    if min_state_duration > 1 and len(x_vals) > 0:
        x_sm = _enforce_min_duration(x_vals, min_len=min_state_duration)
        state_sm = pd.Series(x_sm, index=state_int_filled.index, name="state_int_smooth").astype(int)
    else:
        state_sm = state_int_filled.rename("state_int_smooth")

    # --- Label states into human-readable regimes ---
    feats_aligned = features_std.loc[state_sm.index].copy()
    label_map = label_states_by_stats(feats_aligned, state_sm)
    state_label = state_sm.map(label_map)

    # --- Return all model outputs ---
    return {
        "state_int": state_int_filled,                  # raw state indices
        "state_int_smooth": state_sm,                   # smoothed regime sequence
        "state_label": state_label,                     # mapped labels (e.g., Crisis, Trending)
        "label_map": label_map,                         # integer→string label mapping
        "state_proba": proba_df.dropna(how="all"),      # posterior state probabilities
    }
