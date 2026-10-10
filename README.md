# OncoMark-ML: Transcriptomic Biomarker Discovery Pipeline

## Description
**OncoMark-ML** is an end-to-end machine learning workflow designed to identify key diagnostic biomarker genes from bulk RNA-Seq and transcriptomic data. The pipeline processes high-dimensional gene expression matrices, performs dimensionality reduction to explore cohort variance, and trains a highly robust Random Forest classifier to distinguish between Tumor and Normal patient profiles. Finally, it extracts the most predictive features (genes) using Gini impurity metrics to prioritize targets for downstream clinical validation.

## Key Features & Pipeline Architecture
1. **Data Ingestion & Simulation:** Capable of handling raw transcriptomic count matrices. Includes a built-in synthetic cohort generator simulating 500 patients and 1,000 genes to facilitate immediate testing and pipeline validation.
2. **Dimensionality Reduction (PCA):** Leverages Principal Component Analysis to visualize global expression variance and confirm linear separability between cohorts in 2D space.
3. **Machine Learning Classifier:** Trains a scikit-learn Random Forest algorithm to predict sample conditions. Utilizes stratified train-test splits (70/30) to prevent overfitting and ensure robust generalization.
4. **Biomarker Extraction:** Evaluates and ranks the top 15 most influential genes driving the model's predictions via feature importance scores.
5. **Statistical Evaluation:** Automatically generates comprehensive ML metrics, including Precision, Recall, F1-scores, and Receiver Operating Characteristic (ROC) curves with calculated Area Under the Curve (AUC).

## Repository Structure
OncoMark-ML-Pipeline/
├── scripts/
│   └── ml_biomarker_discovery.py       # Main pipeline script
├── data/
│   └── simulated_expression_cohort.csv # Generated input matrix (Patients x Genes)
├── results/
│   ├── model_evaluation.txt            # Classification report (Precision, Recall, F1)
│   └── plots/
│       ├── PCA_Plot.png                # 2D PCA clustering visualization
│       ├── Feature_Importance.png      # Bar chart of top 15 biomarker genes
│       └── ROC_Curve.png               # ROC-AUC performance curve
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation

## Technologies & Libraries Used
* **Language:** Python 3.x
* **Data Manipulation:** pandas, numpy
* **Machine Learning:** scikit-learn
* **Data Visualization:** matplotlib, seaborn

## Installation & Setup
It is recommended to run this pipeline within an isolated virtual environment to prevent dependency conflicts.

# 1. Clone the repository (if applicable) or navigate to the project directory
cd OncoMark-ML-Pipeline

# 2. Create a virtual environment
python3 -m venv ml_env

# 3. Activate the environment
source ml_env/bin/activate

# 4. Install required dependencies
pip install -r requirements.txt

## Usage
Run the fully automated pipeline with a single command. The script will generate the data, train the model, and output all analytical plots and reports into the results directory.

python scripts/ml_biomarker_discovery.py
