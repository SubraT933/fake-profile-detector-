import os
import sys
import json
import pandas as pd
import numpy as np

datasets_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"

files_and_names = [
    ("strict_prisma_instagram_results.json", "Instagram Benchmark (5,000)"),
    ("strict_prisma_limfadd_results.json", "LIMFADD LinkedIn Benchmark (15,000)"),
    ("results_spammer_genuine.json", "Instagram Spammer vs Genuine (696)"),
    ("results_instafake_fake_real.json", "Instafake: Fake vs Real (1,194)"),
    ("results_instafake_automated.json", "Instafake: Automated Behavior (1,400)")
]

all_dfs = []
for fname, dname in files_and_names:
    fpath = os.path.join(datasets_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df["Dataset_Clean"] = dname
            all_dfs.append(df)
            print(f"Loaded {len(df)} rows from {fname}")

master_df = pd.concat(all_dfs, ignore_index=True)
master_csv_path = os.path.join(datasets_dir, "master_all_datasets_benchmark_summary.csv")
master_json_path = os.path.join(datasets_dir, "master_all_datasets_benchmark_summary.json")

master_df.to_csv(master_csv_path, index=False)
master_df.to_json(master_json_path, orient="records", indent=2)

print(f"\nSaved master summary with {len(master_df)} records to:")
print(f"  {master_csv_path}")
print(f"  {master_json_path}")

# Calculate overall model rankings averaged across all 5 datasets
model_summary = master_df.groupby(["Model", "Category"]).agg(
    Mean_Accuracy=("Accuracy", "mean"),
    Mean_F1=("F1_Score", "mean"),
    Mean_Precision=("Precision", "mean"),
    Mean_Recall=("Recall", "mean"),
    Mean_ROC_AUC=("ROC_AUC", "mean"),
    Mean_Train_Time_s=("Train_Time_s", "mean"),
    Datasets_Evaluated=("Dataset_Clean", "count")
).reset_index()

model_summary = model_summary.sort_values(by=["Mean_F1", "Mean_Accuracy"], ascending=False).reset_index(drop=True)
model_summary["Overall_Rank"] = range(1, len(model_summary) + 1)

print("\n=== OVERALL BENCHMARK RANKING ACROSS ALL 5 DATASETS ===")
cols_to_show = ["Overall_Rank", "Model", "Category", "Mean_Accuracy", "Mean_F1", "Mean_Precision", "Mean_Recall", "Mean_ROC_AUC", "Mean_Train_Time_s"]
print(model_summary[cols_to_show].to_string(index=False))

summary_csv_path = os.path.join(datasets_dir, "overall_model_ranking_summary.csv")
model_summary.to_csv(summary_csv_path, index=False)
