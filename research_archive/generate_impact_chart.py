import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Data from your actual results
models = ['CatBoost', 'LightGBM', 'TabNet', 'TabPFN']
before_scores = [0.9092, 0.9087, 0.7015, 0.8343]
after_scores = [0.9104, 0.9103, 0.8126, 0.5010]

# 2. Setup Plot
x = np.arange(len(models))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Use your project colors: Navy and Gold/Orange
rects1 = ax.bar(x - width/2, before_scores, width, label='Before XAI (10 Features)', color='#003366')
rects2 = ax.bar(x + width/2, after_scores, width, label='After XAI (7 Features)', color='#d4a017')

# 3. Add labels and styling
ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax.set_title('Comparative Impact of XAI-Driven Feature Pruning', fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold', fontsize=11)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.1) # Give space for labels

# 4. Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# 5. Save the image
if not os.path.exists('static'): os.makedirs('static')
plt.tight_layout()
plt.savefig('static/pruning_impact.png', dpi=150)
print("--- SUCCESS: static/pruning_impact.png generated! ---")
plt.show()