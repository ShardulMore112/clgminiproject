import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# 🔄 Load model and tools
with open("model/topic_weakness_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/topic_label_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("model/used_features.json", "r") as f:
    used_features = json.load(f)

# 📊 Load student data
df = pd.read_csv("ml_mcq_student_scores_corrected.csv")
X = df[used_features]

# 🔧 Rebuild labels
threshold = 2
y = []
for _, row in X.iterrows():
    weak_topics = [col for col in X.columns if row[col] < threshold]
    y.append(weak_topics)

Y = mlb.transform(y)

# ✂️ Split test set (same split as train_model.py)
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 🔮 Predict and Evaluate
Y_pred = model.predict(X_test)

print("\n📊 Classification Report:\n")
print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))

print("✅ Accuracy (subset match):", accuracy_score(Y_test, Y_pred))
print("✅ Hamming Loss:", hamming_loss(Y_test, Y_pred))
