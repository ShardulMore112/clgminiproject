import streamlit as st
import json
import pandas as pd
import pickle
from io import BytesIO

# ---------- Load model and metadata ----------
with open("model/topic_weakness_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/topic_label_binarizer.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("model/used_features.json") as f:
    used_columns = json.load(f)

# ---------- Load questions ----------
with open("mcq_question.json") as f:
    raw_questions = json.load(f)

questions = []
for unit in raw_questions["Subject"]["Machine Learning"]["Unit"].values():
    questions.extend(unit["MCQs"])

# ---------- Load resources ----------
with open("resource_map.json") as f:
    resource_map = json.load(f)

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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="ML MCQ Report Card", layout="wide")
st.title("📘 Machine Learning MCQ Test + Report Card")

student_name = st.text_input("👤 Enter your name:")

if student_name:
    st.markdown("### 📝 Take the Test")
    answers = []
    for i, q in enumerate(questions):
        options = list(q["options"].values())
        user_ans = st.radio(f'**Q{i+1}. {q["question"]}**', options, key=f'q{i}')
        answers.append(user_ans)

    if st.button("✅ Submit"):
        topic_scores = {topic: 0 for topic in used_columns}
        correct_count = 0
        incorrect_count = 0

        for i, q in enumerate(questions):
            correct_option = q["answer"]
            correct_answer = q["options"][correct_option]
            topic = q["topic"]
            full_topic = next((col for col in used_columns if col.endswith(topic)), None)

            if answers[i] == correct_answer:
                correct_count += 1
                if full_topic:
                    topic_scores[full_topic] += 1
            else:
                incorrect_count += 1

        df_student = pd.DataFrame([topic_scores])[used_columns]
        predicted_labels = model.predict(df_student)
        predicted_topics = mlb.inverse_transform(predicted_labels)[0]
        resources = recommend_resources(predicted_topics)

        # 🎓 Report Card
        st.markdown("---")
        st.markdown(f"""
        <div style='background:#fff8dc;padding:25px;border-radius:12px;border:1px solid #aaa;'>
            <h2 style='color:#333;'>📄 Report Card</h2>
            <p><strong>Name:</strong> <span style='color:#222;'>{student_name}</span></p>
            <p><strong>Total Questions:</strong> {len(questions)}</p>
            <p><strong style='color:green;'>Correct:</strong> {correct_count} &nbsp;&nbsp; 
               <strong style='color:red;'>Incorrect:</strong> {incorrect_count}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ❌ Weak Topics")
        if predicted_topics:
            for t in predicted_topics:
                t_clean = t.split(" - ")[-1]
                st.markdown(f"<div style='color:#b30000;font-weight:bold;'>🔹 {t_clean}</div>", unsafe_allow_html=True)
        else:
            st.success("🎉 You mastered all topics!")

        st.markdown("### 📚 Recommended Resources")
        for topic, rec in resources.items():
            st.markdown(f"""
            <div style='border-left: 4px solid #0077cc; padding: 10px 15px; margin: 10px 0; background: #f1f9ff;'>
                <strong>📘 {topic}</strong><br>
                🎥 YouTube: <a href="{rec['YouTube']}" target="_blank">{rec['YouTube']}</a><br>
                📘 Course: <a href="{rec['Course']}" target="_blank">{rec['Course']}</a>
            </div>
            """, unsafe_allow_html=True)

        # 📄 Generate downloadable report
        report = [
            f"Name: {student_name}\n",
            f"Correct: {correct_count}, Incorrect: {incorrect_count}\n",
            "Weak Topics:\n",
        ]
        report += [f"- {t.split(' - ')[-1]}\n" for t in predicted_topics]
        report.append("\nResources:\n")
        for topic, rec in resources.items():
            report.append(f"\n{topic}:\n")
            for k, v in rec.items():
                report.append(f"  {k}: {v}\n")

        buffer = BytesIO()
        buffer.write("".join(report).encode())
        buffer.seek(0)

        st.download_button("📥 Download Report", buffer, file_name="ML_ReportCard.txt", mime="text/plain")
