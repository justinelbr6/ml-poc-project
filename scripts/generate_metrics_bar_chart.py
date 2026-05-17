import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
sns.set_theme(style="whitegrid")

# Load metrics
df = pd.read_csv('results/model_metrics.csv')

# Melt the dataframe for seaborn grouped barplot
df_melted = df.melt(id_vars=['model_name'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                    var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Metric', y='Score', hue='model_name', palette='Set2')
plt.title('Comparaison des Performances des Modèles', fontsize=16)
plt.ylim(0.7, 1.05) # zoom in on the interesting part
plt.ylabel('Score')
plt.legend(title='Modèles', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

output_path = 'plots/10_model_metrics_comparison.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
