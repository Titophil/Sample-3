import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from scipy.stats import skew, kurtosis
from sklearn.metrics import classification_report, confusion_matrix


base_dir = "../sample2/a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0"
csv_dir = os.path.join(base_dir, "csv")
info_file = os.path.join(csv_dir, "participants_info.csv")

participants_df = pd.read_csv(info_file)


features= []


for _, row in participants_df.iterrows():
    patient_id = str(row["id_record"]).zfill(4)
    signal_path = os.path.join(csv_dir, f"{patient_id}.csv")
  
    if os.path.exists(signal_path):
      df = pd.read_csv(signal_path)
      
      re = df["RE_1"]
      le = df["LE_1"]

      patient_features = {
                     "id_record": patient_id,
            "re_mean": np.mean(re),
            "re_std": np.std(re),
            "re_min": np.min(re),
            "re_max": np.max(re),
            "re_skew": skew(re),
            "re_kurtosis": kurtosis(re),
            "le_mean": np.mean(le),
            "le_std": np.std(le),
            "le_min": np.min(le),
            "le_max": np.max(le),
            "le_skew": skew(le),
            "le_kurtosis": kurtosis(le),
            "diagnosis1": row["diagnosis1"]
      }

      features.append(patient_features)
features_df = pd.DataFrame(features)
features_df["label"]= features_df["diagnosis1"].apply(lambda x: 0 if x == "Normal" else 1)

X = features_df.drop(["diagnosis1","label","id_record"], axis=1)
y = features_df["label"]




X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

