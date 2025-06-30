import pandas as pd
from sklearn.utils import resample

def download_data(sample_size=100000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    full_df = pd.read_csv(url, header=None, compression='gzip')
    sample = resample(full_df, n_samples=sample_size, random_state=42)
    sample.to_csv('data/higgs_sample.csv', index=False)
    print(f"Saved {sample_size} samples to data/higgs_sample.csv")

if __name__ == "__main__":
    download_data()