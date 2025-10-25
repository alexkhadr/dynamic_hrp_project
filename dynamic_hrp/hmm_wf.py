from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

def _enforce_min_duration(path: np.ndarray, min_len: int = 2) -> np.ndarray:
    """Post-process a discrete state sequence so that any run shorter than min_len is merged with neighbors."""
    if min_len <= 1 or len(path) == 0:
        return path.copy()

    x = path.copy()
    n = len(x)

    # Find initial runs
    runs = []
    start = 0
    for i in range(1, n):
        if x[i] != x[i - 1]:
            runs.append((x[start], start, i - 1))
            start = i
    runs.append((x[start], start, n - 1))

    changed = True
    while changed:
        changed = False
        new_runs = []
        for idx, (state, a, b) in enumerate(runs):
            length = b - a + 1
            if length < min_len:
                left  = runs[idx - 1] if idx > 0 else None
                right = runs[idx + 1] if idx < len(runs) - 1 else None
                candidates = []
                if left is not None:
                    candidates.append(("L", left[0], left[1], left[2], left[2] - left[1] + 1, a - left[2]))
                if right is not None:
                    candidates.append(("R", right[0], right[1], right[2], right[2] - right[1] + 1, right[1] - b))
                if not candidates:
                    continue
                side, new_state, *_ = sorted(candidates, key=lambda t: (-t[4], t[5]))[0]
                x[a:b+1] = new_state
                changed = True
            else:
                new_runs.append((state, a, b))
        if changed:
            # Recompute runs
            runs = []
            start = 0
            for i in range(1, n):
                if x[i] != x[i - 1]:
                    runs.append((x[start], start, i - 1))
                    start = i
            runs.append((x[start], start, n - 1))

    return x

def label_states_by_stats(features: pd.DataFrame, states: pd.Series) -> dict[int, str]:
    """
    Map integer HMM states -> human labels {"Trending","Neutral","Crisis"}
    using means of vol_mean and corr_mean_offdiag per state.
    """
    df = features.copy()
    df["state"] = states.values
    grp = df.groupby("state").mean(numeric_only=True)
    cols = [c for c in ["vol_mean", "corr_mean_offdiag"] if c in grp.columns]
    comp = grp[cols].mean(axis=1) if cols else grp.mean(axis=1)
    order = comp.sort_values().index.tolist()
    mapping: dict[int, str] = {}
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

def fit_hmm_walkforward(
    features_std: pd.DataFrame,
    n_components: int = 3,
    covariance_type: str = "diag",
    refit_every_weeks: int = 4,
    min_train_weeks: int = 104,
    min_state_duration: int = 2,
    random_state: int = 42,
):
    """Walk-forward HMM fit with periodic refits; returns states and probabilities."""
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

    state_int = pd.Series(index=dates, dtype="Int64")
    proba_df = pd.DataFrame(index=dates, columns=[f"p_state_{k}" for k in range(n_components)], dtype=float)

    first_fit_ix = min_train_weeks
    model = None
    last_refit = -10**9

    for t_idx in range(first_fit_ix, len(dates)):
        # Refit on schedule using data up to t-1
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

        # Classify up to current time and take the last state as current
        X_up_to_t = X.iloc[: t_idx + 1].values
        _, vt_path = model.decode(X_up_to_t, algorithm="viterbi")
        state_int.iloc[t_idx] = int(vt_path[-1])

        try:
            post = model.predict_proba(X_up_to_t)
            proba_df.iloc[t_idx, :] = post[-1, :]
        except Exception:
            try:
                framelogprob = model._compute_log_likelihood(X_up_to_t)
                posteriors = model._compute_posteriors(framelogprob)[0]
                proba_df.iloc[t_idx, :] = posteriors[-1, :]
            except Exception:
                pass

    state_int_filled = state_int.bfill().ffill().fillna(0).astype(int)
    x_vals = state_int_filled.values
    if min_state_duration > 1 and len(x_vals) > 0:
        x_sm = _enforce_min_duration(x_vals, min_len=min_state_duration)
        state_sm = pd.Series(x_sm, index=state_int_filled.index, name="state_int_smooth").astype(int)
    else:
        state_sm = state_int_filled.rename("state_int_smooth")

    feats_aligned = features_std.loc[state_sm.index].copy()
    label_map = label_states_by_stats(feats_aligned, state_sm)
    state_label = state_sm.map(label_map)

    return {
        "state_int": state_int_filled,
        "state_int_smooth": state_sm,
        "state_label": state_label,
        "label_map": label_map,
        "state_proba": proba_df.dropna(how="all"),
    }
