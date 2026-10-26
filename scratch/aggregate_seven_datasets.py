import os
import shutil
import json
import pandas as pd

datasets_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"

# Copy results_instafake_fake_real.json to results_user_custom_dataset.json if needed
src = os.path.join(datasets_dir, "results_instafake_fake_real.json")
dst = os.path.join(datasets_dir, "results_user_custom_dataset.json")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy(src, dst)
    print("Copied fake insta results to results_user_custom_dataset.json")

seven_datasets = [
    ("results_spammer_genuine.json", "Dataset 1 (Instagram Spammer & Genuine)", 696),
    ("results_instafake_fake_real.json", "Fake Insta (Instafake Fake vs Real)", 1194),
    ("results_instafake_automated.json", "Instafake Automated (Automated Behavior)", 1400),
    ("strict_prisma_instagram_results.json", "Instagram 5K (Instagram 5,000 Profiles)", 5000),
    ("strict_prisma_limfadd_results.json", "LIMFADD (LinkedIn Multi-attribute)", 15000),
    ("results_cresci_2015.json", "Cresci-2015 / Cersi (Twitter Bot Benchmark)", 5301),
    ("results_twibot_22.json", "TwiBot-22 / Twibot (Twitter Bot Benchmark)", 10000)
]

dfs = []
for fname, pretty_name, n_samples in seven_datasets:
    fpath = os.path.join(datasets_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df["Dataset_Clean"] = pretty_name
            df["Dataset_Samples"] = n_samples
            dfs.append(df)
            print(f"Loaded {len(df)} algorithms for: {pretty_name}")

master_df = pd.concat(dfs, ignore_index=True)
master_csv = os.path.join(datasets_dir, "master_seven_datasets_benchmark_summary.csv")
master_json = os.path.join(datasets_dir, "master_seven_datasets_benchmark_summary.json")
master_df.to_csv(master_csv, index=False)
master_df.to_json(master_json, orient="records", indent=2)

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
summary_csv = os.path.join(datasets_dir, "seven_datasets_model_ranking_summary.csv")
model_summary.to_csv(summary_csv, index=False)

print(f"\nSuccessfully loaded ALL {len(dfs)} DATASETS! Total benchmark rows: {len(master_df)}")
cols_to_show = ["Overall_Rank", "Model", "Category", "Mean_Accuracy", "Mean_F1", "Mean_Precision", "Mean_Recall", "Mean_ROC_AUC", "Mean_Train_Time_s"]
print(model_summary[cols_to_show].to_string(index=False))
