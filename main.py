import time
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.preprocessing import preprocess_data
from src.feature_selection import select_features
from src.modeling import get_models, nested_cv
from src.evaluation import evaluate_model

def main():
    # Create output directories
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    
    # Preprocessing
    print("Preprocessing data...")
    X, y = preprocess_data()
    
    # Feature Selection
    print("Selecting features...")
    X_selected, selected_idx = select_features(X, y, method='anova', k=15)
    print(f"Selected features: {selected_idx}")
    
    # Modeling
    models = get_models()
    all_results = {}
    
    for name, config in tqdm(models.items(), desc="Training models"):
        print(f"\nTraining {name} model...")
        start = time.time()
        
        # Nested CV
        cv_results = nested_cv(config['model'], config['params'], X_selected, y)
        
        # Evaluation
        metrics_list = []
        for i, res in enumerate(cv_results):
            metrics = evaluate_model(res['model'], *res['test_set'])
            metrics['best_params'] = res['best_params']
            metrics_list.append(metrics)
            
            # Save ROC curve
            plt.savefig(f"outputs/figures/{name}_roc_fold_{i+1}.png")
            plt.close()
        
        # Save results
        results_df = pd.DataFrame(metrics_list)
        results_df.to_csv(f"outputs/results/{name}_metrics.csv", index=False)
        all_results[name] = results_df
        
        print(f"{name} completed in {time.time()-start:.2f} seconds")
    
    # Generate final report
    generate_report(all_results)
    print("Pipeline completed successfully!")

def generate_report(results):
    # Aggregate and compare results
    final_metrics = {}
    for name, df in results.items():
        final_metrics[name] = {
            'Accuracy': df['Accuracy'].mean(),
            'Precision': df['Precision'].mean(),
            'Recall': df['Recall'].mean(),
            'F1': df['F1'].mean(),
            'ROC-AUC': df['ROC-AUC'].mean()
        }
    
    # Create summary table
    summary_df = pd.DataFrame(final_metrics).T
    summary_df.to_csv("outputs/results/summary_metrics.csv")
    
    # Identify best model
    best_model = summary_df['ROC-AUC'].idxmax()
    print(f"\nBest model: {best_model} (AUC: {summary_df.loc[best_model, 'ROC-AUC']:.4f})")

if __name__ == "__main__":
    main()