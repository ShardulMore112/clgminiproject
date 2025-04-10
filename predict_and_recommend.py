import pandas as pd
import json
import pickle

# 🔄 Load model and binarizer
with open("model/topic_weakness_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/topic_label_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("model/used_features.json") as f:
    used_columns = json.load(f)

# 🧾 Load student data and resource map
df = pd.read_csv("ml_mcq_student_scores_corrected.csv")
X = df[used_columns]

with open("resource_map.json") as f:
    resource_map = json.load(f)

# 🎯 Recommend resources
def recommend_resources(weak_topics):
    recommendations = {}
    for topic in weak_topics:
        topic_clean = topic.split(" - ", 1)[-1]
        found = False
        for subject in resource_map["Subject"].values():
            for unit in subject["Unit"].values():
                if topic_clean in unit:
                    recommendations[topic_clean] = unit[topic_clean]
                    found = True
                    break
            if found:
                break
        if not found:
            recommendations[topic_clean] = {"YouTube": "N/A", "Course": "N/A"}
    return recommendations

# 🎯 Predict
def predict_weak_topics(student_idx=0):
    student_id = df.iloc[student_idx]["Student ID"]
    new_student = X.iloc[[student_idx]]
    predicted_labels = model.predict(new_student)
    predicted_topics = mlb.inverse_transform(predicted_labels)[0]
    return student_id, predicted_topics

student_id, predicted_topics = predict_weak_topics(0)
resources = recommend_resources(predicted_topics)

print(f"\n📘 Weak Topics for {student_id}:\n{predicted_topics or 'None 🎉'}")
print("\n📚 Recommended Resources:")
for topic, rec in resources.items():
    print(f"\n🔸 {topic}")
    for k, v in rec.items():
        print(f"   {k}: {v}")
