<font face="Arial" color="blue" size="20">HiggsML Pipeline: Feature Selection & Hyperparameter Optimization</font>

<ins>Project Overview</ins>

This project implements a complete machine learning pipeline for the HIGGS dataset, focusing on feature selection and hyperparameter optimization. The HIGGS dataset contains 11 million samples of high-energy physics data, with 28 features distinguishing between Higgs boson signals (class 1) and background noise (class 0). For computational efficiency, we used a random sample of 100,000 observations.

https://img.shields.io/badge/Python-3.11-blue.svg


![Logo full](https://github.com/user-attachments/assets/727625d6-0fcf-4653-a3e0-f9b44bf581cf)


Key Results

Model Performance Comparison

![image](https://github.com/user-attachments/assets/fed29a17-83dc-4969-8c1b-5421093067a6)

ROC Curve Comparison




Feature Importance





Project Structure

![image](https://github.com/user-attachments/assets/d93ace03-59b1-463a-9e19-3a37a7bf34ed)


Methodology
1. Data Preprocessing
    Outlier Handling: IQR method to cap outliers at 1.5 times the interquartile range

    Feature Scaling: MinMaxScaler to normalize features to [0,1] range

2. Feature Selection
    Used ANOVA F-test to select top 15 most important features

    Selected features: [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21]

3. Modeling
    Implemented nested cross-validation:

        Outer Loop: 5-fold for performance evaluation

        Inner Loop: 3-fold for hyperparameter tuning

Tested four models:

  K-Nearest Neighbors (KNN)

  Support Vector Machine (SVM)

  Multi-Layer Perceptron (MLP)

  XGBoost

4. Evaluation
    Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC

    ROC curves for each model comparison

    Comprehensive performance analysis

    Installation & Setup
   
1. Clone the repository:

        git clone https://github.com/yourusername/higgs-ml-project.git

        cd higgs-ml-project

2. Create and activate virtual environment:

        python -m venv higgs-env

        # Windows:

        higgs-env\Scripts\activate

        # Linux/Mac:

        source higgs-env/bin/activate

3. Install dependencies:

       pip install -r requirements.txt

Running the Pipeline

1. Download the dataset (100,000 samples):

       python data/download_higgs.py

2. Execute the full pipeline:

       python main.py

3. Generate the final report:

       python latex_docker.py

Key Findings

1. XGBoost Dominates: Achieved best performance across all metrics (AUC: 0.812)
  
2. Feature Selection Effective: Reduced dimensionality by 46% (28 → 15 features) with minimal performance loss

3. Physics Features Matter: Top features relate to particle kinematics and angular relationships

4. Hyperparameter Sensitivity: MLP showed most sensitivity to hyperparameter choices

    Best Model: XGBoost

    Optimal Hyperparameters

      ![image](https://github.com/user-attachments/assets/6750de73-9272-4888-a701-ee7b6f7d44bd)


Feature Importance



Conclusion

The XGBoost model demonstrated superior performance in distinguishing Higgs boson signals from background noise. The pipeline successfully:

1. Handled large-scale physics data efficiently

2. Identified the most discriminative features

3. Optimized model hyperparameters through nested cross-validation

4. Provided comprehensive performance evaluation

This implementation provides a robust template for similar high-energy physics classification tasks.

License

This project is licensed under the NeuronWorks AI License.
