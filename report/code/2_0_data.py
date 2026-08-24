import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Four numeric features: 3 informative, 1 redundant (a linear combination of them)
X_num, y = make_classification(
    n_samples=600,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    class_sep=0.9,         # how far apart the two class clusters sit
    weights=[0.68, 0.32],  # class imbalance: 68% negative, 32% positive
    flip_y=0.03,           # 3% of labels randomly flipped (label noise)
    random_state=21
)

df = pd.DataFrame(X_num, columns=["x1", "x2", "x3", "x4"])
rng = np.random.default_rng(21)
df["region"] = rng.choice(["north", "south", "east"], size=len(df))  # nominal, 3 levels
df["device"] = rng.choice(["mobile", "desktop"], size=len(df))       # nominal, 2 levels
df.loc[rng.choice(len(df), 25, replace=False), "x3"] = np.nan        # 25 missing values
