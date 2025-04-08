# 🧠 AI-Based Test Generator & Learning Recommendation System

This project is an AI-powered personalized testing and recommendation platform for students. It uses RAG (Retrieval-Augmented Generation) to generate subject-wise tests from notes and evaluates student performance to identify weak areas. Based on these weak topics, it recommends study resources such as PDFs, videos, and courses to help improve understanding.

---

## 📌 Features

- ✅ Automatically generate MCQ tests using provided notes (RAG)
- ✅ 60-question test covering 3 subjects (20 per subject)
- ✅ Evaluate student answers and identify weak units/topics
- ✅ Rule-based or ML-based topic weakness detection
- ✅ Personalized resource recommendations (PDFs, videos, courses)
- ✅ Visual feedback on subject/unit-wise performance
- ✅ Streamlit frontend for user interaction and result visualization

---

## 📁 Subjects Covered

- **Mathematics**
  - Unit 1: Algebra
  - Unit 2: Calculus

- **Physics**
  - Unit 1: Mechanics
  - Unit 2: Electricity

- **Chemistry**
  - Unit 1: Organic Chemistry
  - Unit 2: Physical Chemistry

*(These can be customized in your own dataset.)*

---

## ⚙️ How It Works

1. **Test Generation:**  
   RAG system creates questions from notes you upload.

2. **Student Attempts Test:**  
   Streamlit UI shows 60 questions – 20 per subject.

3. **Evaluation Engine:**  
   Answers are scored per topic (subject, unit, and topic level).

4. **Weak Topic Identification:**  
   - Option 1: Rule-based (e.g., <50% correct = weak)
   - Option 2: ML-based (using historical performance)

5. **Recommendations:**  
   For weak topics, the system suggests:
   - 📄 PDF Notes
   - 🎥 Videos
   - 📘 Online Courses

---

## 🧪 Tech Stack

- 🧠 **RAG Model**: For test generation
- 🐍 **Python**: Core logic
- 📊 **Pandas, scikit-learn**: Evaluation + ML classification
- 🎨 **Streamlit**: Frontend UI
- 📁 **JSON**: Resource mapping

