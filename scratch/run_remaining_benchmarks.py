import os
import sys
import json
import pandas as pd

sys.path.insert(0, r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed")
from benchmark_engine import benchmark_dataset

datasets_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"

benchmarks = [
    ("Instagram_Spammer_Genuine.csv", "Instagram Spammer & Genuine (696 profiles)", "results_spammer_genuine.json"),
    ("Instafake_Fake_vs_Real.csv", "Instafake: Fake vs Real (1,194 profiles)", "results_instafake_fake_real.json"),
    ("Instafake_Automated_Behavior.csv", "Instafake: Automated Behavior (1,400 profiles)", "results_instafake_automated.json")
]

for filename, pretty_name, out_json in benchmarks:
    csv_path = os.path.join(datasets_dir, filename)
    print(f"\n>>> Running Benchmark for: {pretty_name} <<<", flush=True)
    summary_df, full_df = benchmark_dataset(csv_path, dataset_name=pretty_name)
    
    out_path = os.path.join(datasets_dir, out_json)
    full_df.to_json(out_path, orient="records", indent=2)
    print(f"Saved results to {out_path}", flush=True)

print("\nALL BENCHMARKS COMPLETED SUCCESSFULLY!", flush=True)
