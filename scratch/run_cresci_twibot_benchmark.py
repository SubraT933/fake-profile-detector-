import os
import sys
import json
import pandas as pd

sys.path.insert(0, r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed")
from benchmark_engine import benchmark_dataset

target_dir = r"c:\Users\subra\OneDrive\Desktop\fake_profile_user_input_website_fixed\datasets"

# 1. Benchmark Cresci-2015
p_cresci = os.path.join(target_dir, "Cresci_2015.csv")
print("=== Running Benchmark for Cresci-2015 ===", flush=True)
s_df_cresci, full_cresci = benchmark_dataset(p_cresci, dataset_name="Cresci-2015 (5,301 profiles)")
out_cresci_json = os.path.join(target_dir, "results_cresci_2015.json")
full_cresci.to_json(out_cresci_json, orient="records", indent=2)
print(f"Saved Cresci results to {out_cresci_json}", flush=True)

# 2. Benchmark TwiBot-22
p_twi = os.path.join(target_dir, "TwiBot_22.csv")
print("\n=== Running Benchmark for TwiBot-22 ===", flush=True)
s_df_twi, full_twi = benchmark_dataset(p_twi, dataset_name="TwiBot-22 (10,000 profiles)")
out_twi_json = os.path.join(target_dir, "results_twibot_22.json")
full_twi.to_json(out_twi_json, orient="records", indent=2)
print(f"Saved TwiBot results to {out_twi_json}", flush=True)

print("\nALL CRESCI AND TWIBOT BENCHMARKS COMPLETED!", flush=True)
