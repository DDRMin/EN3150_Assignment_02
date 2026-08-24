from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = ["x1", "x2", "x3", "x4"]     # numerical features
cat_cols = ["region", "device"]         # categorical features

X = df[num_cols + cat_cols]

# Stratified 80:20 split, so both sets keep the same class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# Numerical: fill missing with the median, then standardize
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# Categorical: fill missing with the mode, then one-hot encode
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Apply each sub-pipeline to its own set of columns
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# Full pipeline: preprocessing + classifier
clf = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

clf.fit(X_train, y_train)

print(f"Train size: {X_train.shape[0]}, test size: {X_test.shape[0]}")
print(f"Positive class share - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
print(f"Missing x3 values    - train: {X_train['x3'].isna().sum()}, "
      f"test: {X_test['x3'].isna().sum()}")

# Column names after one-hot encoding
print("\nFeatures after preprocessing:")
print(list(clf.named_steps["pre"].get_feature_names_out()))

print(f"\nTrain accuracy: {clf.score(X_train, y_train):.4f}")
print(f"Test accuracy : {clf.score(X_test, y_test):.4f}")
