import streamlit as st
import json

# Load quiz data
with open("CN_Unit1.json") as f:
    quiz_data = json.load(f)

st.title(quiz_data.get("title", "Computer Networks Quiz"))

# Initialize session state
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "score" not in st.session_state:
    st.session_state.score = 0

total_questions = len(quiz_data["quiz"])
q_index = max(0, min(st.session_state.current_q, total_questions - 1))
st.session_state.current_q = q_index  # Keep it in valid range
question = quiz_data["quiz"][q_index]

# Show current question
st.markdown(f"### Question {q_index + 1} of {total_questions}")
st.markdown(f"**{question['question']}**")

# Options
selected_option = st.session_state.answers.get(q_index)
try:
    option = st.radio(
        "Choose one:",
        options=list(question["options"]),
        format_func=lambda x: f"{x}) {question['options'][x]}",
        index=list(question["options"]).index(selected_option) if selected_option else -1,
        key=f"q_{q_index}"
    )
except st.errors.StreamlitAPIException:
    option = st.radio(
        "Choose one:",
        options=list(question["options"]),
        format_func=lambda x: f"{x}) {question['options'][x]}",
        key=f"q_{q_index}"
    )

# Save answer
if option:
    st.session_state.answers[q_index] = option

# Navigation buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Previous", disabled=q_index == 0):
        st.session_state.current_q -= 1
with col2:
    if st.button("Next", disabled=q_index == total_questions - 1):
        st.session_state.current_q += 1
with col3:
    if q_index == total_questions - 1:
        if st.button("Submit Quiz"):
            score = 0
            for idx, q in enumerate(quiz_data["quiz"]):
                if st.session_state.answers.get(idx) == q["answer"]:
                    score += 1
            st.session_state.score = score
            st.session_state.submitted = True

# Show results
if st.session_state.submitted:
    st.markdown("---")
    st.subheader(f"🎉 Your Final Score: {st.session_state.score} / {total_questions}")
    for idx, q in enumerate(quiz_data["quiz"]):
        user_ans = st.session_state.answers.get(idx, "Not Answered")
        correct = q["answer"]
        result = "✅ Correct" if user_ans == correct else f"❌ Wrong (Correct: {correct})"
        st.markdown(f"**Q{idx+1}. {q['question']}**")
        if user_ans in q["options"]:
            st.markdown(f"Your Answer: {user_ans}) {q['options'][user_ans]}")
        else:
            st.markdown("Your Answer: Not Answered")
        st.markdown(result)
        st.markdown("---")
