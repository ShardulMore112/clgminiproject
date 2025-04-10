import pandas as pd
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# 🔧 Ensure model folder exists
os.makedirs("model", exist_ok=True)

# 📥 Load data
df = pd.read_csv("ml_mcq_student_scores_corrected.csv")

# 🎯 Extract features (topics) and prepare labels
X = df.drop(columns=["Student ID", "Total Score"])
X = X.loc[:, X.nunique() > 1]  # Remove constant columns

# 🎯 Label generation: weak if score < 2
threshold = 2
y = []
for _, row in X.iterrows():
    weak_topics = [col for col in X.columns if row[col] < threshold]
    y.append(weak_topics)

# 🔄 MultiLabelBinarizer for multi-label classification
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# 🔬 Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🧠 Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# 💾 Save model, binarizer, and features
with open("model/topic_weakness_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/topic_label_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

with open("model/used_features.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Model, binarizer, and features saved!")
