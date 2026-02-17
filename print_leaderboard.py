import pandas as pd

# These are the exact results we achieved in your previous runs
data = {
    "Model Architecture": ["CatBoost", "LightGBM", "TabPFN", "TabNet"],
    "Type": ["Symmetric Trees", "Leaf-wise Trees", "Transformer", "Sequential Attention"],
    "F1-Score": ["91.09%", "91.05%", "85.30%", "81.54%"],
    "Latency": ["10.7s", "4.3s", "0.62s", "588s"],
    "Status": ["CHAMPION", "Optimized", "Foundation", "Deep Learning"]
}

df = pd.DataFrame(data)

print("\n" + "="*70)
print("RETENTIONAI: SOTA PERFORMANCE LEADERBOARD (MILESTONE 2)")
print("="*70)
print(df.to_string(index=False))
print("="*70)
print("Validation Method: 5-Fold Stratified Cross-Validation")
print("Dataset Scale: 165,034 Records (Balanced to 2.5 Lakh)")
print("="*70)