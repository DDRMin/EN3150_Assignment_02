import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

indexno = 220525   # index number, used as the fixed random seed

rng = np.random.default_rng(indexno)
X = rng.uniform(-3, 3, size=(160, 1))                   # 160 samples, one feature in [-3, 3]
y = 2.5 + 1.8 * X[:, 0] + rng.normal(0, 0.8, size=160)  # true line y = 2.5 + 1.8x plus N(0, 0.8) noise

# Add large errors to 12 observations.
outlier_idx = rng.choice(len(y), size=12, replace=False)  # rows chosen to be corrupted
y[outlier_idx] += rng.uniform(10, 15, size=12)            # shift them up by 10-15 (vertical outliers)
