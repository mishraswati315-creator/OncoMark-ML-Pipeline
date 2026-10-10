import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, classification_report

# Ensure output directories exist
os.makedirs('results/plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("[1/4] Generating synthetic RNA-Seq cohort data...")
# Simulate 500 patients, 1000 genes, where 20 genes are true biomarkers
X, y = make_classification(n_samples=500, n_features=1000, n_informative=20, 
                           n_redundant=5, random_state=42)
gene_names = [f"GENE_{i+1}" for i in range(1000)]
df = pd.DataFrame(X, columns=gene_names)
df['Condition'] = np.where(y == 1, 'Tumor', 'Normal')
df.to_csv('data/simulated_expression_cohort.csv', index=False)

print("[2/4] Performing Dimensionality Reduction (PCA)...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=df['Condition'], palette='Set1')
plt.title('PCA of Transcriptomic Profiles')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.savefig('results/plots/PCA_Plot.png', dpi=300)
plt.close()

print("[3/4] Training Random Forest Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Save classification report
report = classification_report(y_test, y_pred)
with open('results/model_evaluation.txt', 'w') as f:
    f.write("Random Forest Classification Report\n")
    f.write("===================================\n")
    f.write(report)

print("[4/4] Generating Feature Importance & ROC Curves...")
# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15] # Top 15 genes
top_genes = [gene_names[i] for i in indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=top_genes, palette='viridis')
plt.title('Top 15 Biomarker Genes (Random Forest Feature Importance)')
plt.xlabel('Importance Score')
plt.savefig('results/plots/Feature_Importance.png', dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('results/plots/ROC_Curve.png', dpi=300)
plt.close()

print("Pipeline completed successfully! All plots and data saved in 'results/' and 'data/'.")
