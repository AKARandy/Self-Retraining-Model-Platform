"""Create a drifted copy of the training CSV: inflation-style shift on numeric
features. Registered as dataset version 2 to prove versioned feature sets."""
import pandas as pd

SRC = "data/raw/inbox/kaggle_train.csv"
DST = "data/raw/inbox/kaggle_train_drifted.csv"

df = pd.read_csv(SRC)

# simulate a market shift
df["GrLivArea"] = (df["GrLivArea"] * 1.25).round(0)      # bigger living areas
df["YearBuilt"] = df["YearBuilt"] + 15                    # newer builds
df["OverallQual"] = (df["OverallQual"] + 1).clip(upper=10)
df["LotArea"] = (df["LotArea"] * 1.15).round(0)
# hold the target roughly flat so the shift is purely in the feature space
df["SalePrice"] = (df["SalePrice"] * 0.93).round(0)

df.to_csv(DST, index=False)
print("wrote", DST, "-", df.shape[0], "rows; GrLivArea mean:", round(df['GrLivArea'].mean()), "(orig:", round(pd.read_csv(SRC)['GrLivArea'].mean()), ")")
