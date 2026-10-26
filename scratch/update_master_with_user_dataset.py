import os
import json
import pandas as pd

datasets_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"

# We have:
# 1. strict_prisma_instagram_results.json (Instagram 5K)
# 2. strict_prisma_limfadd_results.json (LIMFADD 15K)
# 3. results_spammer_genuine.json (Instagram Spammer vs Genuine 696)
# 4. results_instafake_automated.json (Instafake Automated 1,400)
# 5. user_custom_dataset benchmark results

# First save the user_custom_dataset results to a json file
from benchmark_engine import benchmark_dataset

user_csv = os.path.join(datasets_dir, "user_custom_dataset.csv")
summary_df, full_df = benchmark_dataset(user_csv, dataset_name="User Provided Dataset (1,194 profiles)")
user_json = os.path.join(datasets_dir, "results_user_custom_dataset.json")
full_df.to_json(user_json, orient="records", indent=2)

files_and_names = [
    ("strict_prisma_instagram_results.json", "Instagram Benchmark (5,000)"),
    ("strict_prisma_limfadd_results.json", "LIMFADD LinkedIn Benchmark (15,000)"),
    ("results_spammer_genuine.json", "Instagram Spammer vs Genuine (696)"),
    ("results_instafake_automated.json", "Instafake: Automated Behavior (1,400)"),
    ("results_user_custom_dataset.json", "User Provided Dataset (1,194)")
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

master_df = pd.concat(all_dfs, ignore_index=True)
master_csv_path = os.path.join(datasets_dir, "master_all_datasets_benchmark_summary.csv")
master_json_path = os.path.join(datasets_dir, "master_all_datasets_benchmark_summary.json")

master_df.to_csv(master_csv_path, index=False)
master_df.to_json(master_json_path, orient="records", indent=2)

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
summary_csv_path = os.path.join(datasets_dir, "overall_model_ranking_summary.csv")
model_summary.to_csv(summary_csv_path, index=False)

print("SUMMARY UPDATE COMPLETE")
print(model_summary.to_string(index=False))
