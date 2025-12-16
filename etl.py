
import numpy as np
import pandas as pd

def etl_transform_for_pipeline(X, train_types, train_columns, min_pol, max_pol, seed):
    X = X.copy()
    rng = np.random.default_rng(seed)

    if "number_of_policies" in X.columns:
        s = X["number_of_policies"]
        mask = s.isna()
        if mask.any():
            X.loc[mask, "number_of_policies"] = rng.integers(min_pol, max_pol + 1, size=int(mask.sum()))
        X["number_of_policies"] = X["number_of_policies"].astype(int)

    if "creature_type" in X.columns:
        s = X["creature_type"]
        mask = s.isna()
        if mask.any():
            X.loc[mask, "creature_type"] = rng.choice(train_types, size=int(mask.sum()), replace=True)

    X = pd.get_dummies(
        X,
        columns=[c for c in ["creature_type", "flight_status"] if c in X.columns],
        drop_first=True,
        dtype=int
    )

    X = X.reindex(columns=train_columns, fill_value=0)

    return X.values.astype(np.float32)
