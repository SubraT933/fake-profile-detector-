import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Test dataset loader
from benchmark_engine import prepare_any_dataset, evaluate_metrics

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Instagram.csv")
df = pd.read_csv(data_path)
X, y, f_names, t_col = prepare_any_dataset(df)
print(f"Data prepared: {X.shape}, target={t_col}, real={np.sum(y==0)}, fake={np.sum(y==1)}")
print("Verification complete!")
