import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kurtosis, skew

# NASA Dataset 1 folder
folder = Path(r"C:\Users\dhars\OneDrive\Desktop\project\nasa ims\1st_test\1st_test")

# Folder where this script is located (test1 folder)
current_folder = Path(__file__).parent

# Lists to store features
rms_values = []
peak_values = []
std_values = []
kurtosis_values = []
skewness_values = []

# to read all files
files = sorted(folder.iterdir())

for file in files:

    # vibration file
    df = pd.read_csv(
        file,
        sep=r"\s+",
        header=None
    )

    # Bearing 3 (failed bearing in Dataset 1)
    signal = df[4]

    # Features extraction
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    std = np.std(signal)
    kurt = kurtosis(signal)
    sk = skew(signal)

    # Store values
    rms_values.append(rms)
    peak_values.append(peak)
    std_values.append(std)
    kurtosis_values.append(kurt)
    skewness_values.append(sk)

print("Total files processed:", len(files))

# Create dataframe
features_df = pd.DataFrame({
    "RMS": rms_values,
    "Peak": peak_values,
    "STD": std_values,
    "Kurtosis": kurtosis_values,
    "Skewness": skewness_values
})

warning_hours = 36
warning_files = int((warning_hours * 60) / 10)
labels = []

for i in range(len(features_df)):
    if i < len(features_df) - warning_files:
        labels.append(0)  # Healthy
    else:
        labels.append(1)  # Failure expected within 36 hours 

features_df["Label"] = labels

# Save labeled dataset
features_df.to_csv(
    current_folder / "bearing3_features_labeled.csv",
    index=False
)

print("Labeled dataset saved successfully.")

print("\nLabel Distribution:")
print(features_df["Label"].value_counts())

# Plot features
plt.figure(figsize=(12, 6))

plt.plot(rms_values, label="RMS")
plt.plot(peak_values, label="Peak")
plt.plot(std_values, label="STD")
plt.plot(kurtosis_values, label="Kurtosis")
plt.plot(skewness_values, label="Skewness")

plt.xlabel("Sample Number")
plt.ylabel("Feature Value")
plt.title("Bearing 3 Feature Analysis")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig(
    current_folder / "bearing3_features.png"
)

plt.show()