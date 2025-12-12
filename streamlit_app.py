import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import random

from topic_blocks import topic_blocks

# =========================
# Пути к файлам (в репо рядом)
# =========================
DATA_PATH = Path("math_tasks.csv")
MODEL_PATH = Path("nlp_model (1).pkl")          
PROGRESS_PATH = Path("progress.json")
ATTEMPTS_PATH = Path("attempts_log.csv")

# =========================
# XP / уровни
# =========================
xp_rewards = {"easy": 5, "medium": 10, "hard": 20}
level_thresholds = {1: 0, 2: 50, 3: 120, 4: 250, 5: 500, 6: 1000}

# =========================
# Утилиты
# =========================
def clean_math_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

def ensure_files():
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("{}", encoding="utf-8")

    if not ATTEMPTS_PATH.exists():
        pd.DataFrame(
            columns=["ts", "student_id", "topic", "difficulty", "correct", "task_text"]
        ).to_csv(ATTEMPTS_PATH, index=False, encoding="utf-8-sig")

def load_progress() -> dict:
    ensure_files()
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))

def save_progress(progress: dict):
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

def get_student_progress(student_id: str) -> dict:
    progress = load_progress()
    if student_id not in progress:
        progress[student_id] = {"xp": 0, "level": 1, "streak": 0}
        save_progress(progress)
    return progress[student_id]

def add_xp(student_id: str, difficulty: str):
    difficulty = (difficulty or "medium").lower()
    reward = xp_rewards.get(difficulty, 10)

    progress = load_progress()
    if student_id not in progress:
        progress[student_id] = {"xp": 0, "level": 1, "streak": 0}

    progress[student_id]["xp"] += reward
    progress[student_id]["streak"] += 1

    xp = progress[student_id]["xp"]
    new_level = 1
    for lvl, req in sorted(level_thresholds.items()):
        if xp >= req:
            new_level = lvl
    progress[student_id]["level"] = new_level

    save_progress(progress)
    return progress[student_id], reward

def log_attempt(student_id: str, topic: str, difficulty: str, correct: int, task_text: str):
    ensure_files()
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "student_id": str(student_id),
        "topic": str(topic),
        "difficulty": (difficulty or "medium").lower(),
        "correct": int(correct),
        "task_text": str(task_text),
    }
    df = pd.read_csv(ATTEMPTS_PATH, encoding="utf-8-sig")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(ATTEMPTS_PATH, index=False, encoding="utf-8-sig")

def pick_task(df, topic_col, text_col, topic, difficulty):
    sub = df[df[topic_col] == topic].copy()
    if sub.empty:
        return None

    if "difficulty" in sub.columns:
        sub["difficulty"] = sub["difficulty"].astype(str).str.lower()
        dsub = sub[sub["difficulty"] == difficulty.lower()]
        if not dsub.empty:
            sub = dsub

    return sub.sample(1, random_state=random.randint(0, 10_000))[text_col].values[0]

def show_topic_block(topic_key: str):
    info = topic_blocks.get(topic_key)
    if not info:
        st.warning("Для этой темы нет topic_blocks.")
        return

    with st.expander("📖 Определение", expanded=False):
        st.write(info.get("definition", ""))

    with st.expander("📘 Конспект", expanded=False):
        st.write(info.get("summary", ""))

    if info.get("formulas"):
        with st.expander("🧾 Формулы", expanded=False):
            for f in info["formulas"]:
                st.write(f"- {f}")

    if info.get("example"):
        with st.expander("💡 Пример", expanded=False):
            st.write(info["example"])

    if info.get("youtube"):
        with st.expander("🎥 YouTube", expanded=False):
            y = info["youtube"]
            if isinstance(y, list):
                for link in y:
                    st.write(link)
            else:
                st.write(y)

def get_hints_for_text(text: str, model, topic_blocks: dict):
    cleaned = clean_math_text(text)
    predicted_topic = model.predict([cleaned])[0]

    info = topic_blocks.get(predicted_topic, {})
    hint1 = info.get("hint1") or "Подумай, какое правило/формула подходит."
    hint2 = info.get("hint2") or "Разбей на шаги: дано → найти → формула → подстановка."
    return predicted_topic, hint1, hint2

# =========================
# Загрузка данных/модели
# =========================
@st.cache_data
def load_tasks():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if "topic_clean" in df.columns:
        topic_col = "topic_clean"
    elif "topic" in df.columns:
        topic_col = "topic"
    else:
        raise ValueError("Нет колонки topic_clean или topic в math_tasks.csv")

    text_col = "task_text" if "task_text" in df.columns else df.columns[0]

    if "difficulty" not in df.columns:
        df["difficulty"] = "medium"

    return df, topic_col, text_col

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def build_attempts_df():
    ensure_files()
    df = pd.read_csv(ATTEMPTS_PATH, encoding="utf-8-sig")
    if not df.empty:
        df["student_id"] = df["student_id"].astype(str)
        df["topic"] = df["topic"].astype(str)
        df["difficulty"] = df["difficulty"].astype(str).str.lower()
        df["correct"] = df["correct"].astype(int)
    return df

def build_pivot_table_from_attempts(df):
    if df is None or df.empty:
        return None
    return df.pivot_table(index="student_id", columns="topic", values="correct", aggfunc="mean")

def recommend_for_student_streamlit(student_id: str, tasks_df, topic_col, text_col):
    attempts_df = build_attempts_df()
    if attempts_df.empty:
        st.info("Пока нет попыток в attempts_log.csv. Перейди в «Учитель» и нажми «Засчитать».")
        return

    pivot_table = build_pivot_table_from_attempts(attempts_df)
    if pivot_table is None or student_id not in pivot_table.index:
        st.warning("По этому student_id пока нет данных.")
        return

    row = pivot_table.loc[student_id].dropna()
    if row.empty:
        st.warning("Недостаточно данных для рекомендаций.")
        return

    weak_topics = row[row < 0.5].sort_values().index.tolist()
    if not weak_topics:
        st.success("Слабых тем не найдено (точность по всем темам ≥ 0.5).")
        return

    st.write(f"❌ Слабые темы (точность < 0.5): **{weak_topics}**")

    for topic in weak_topics:
        st.markdown("---")
        st.subheader(f"📌 {topic}")

        if topic in topic_blocks:
            show_topic_block(topic)
        else:
            st.info(f"Для темы '{topic}' нет блока в topic_blocks.py")

        st.write("🧠 Практика (1 задача на уровень):")
        for diff in ["easy", "medium", "hard"]:
            t = pick_task(tasks_df, topic_col, text_col, topic, diff)
            if t:
                st.write(f"**{diff.upper()}**: {t}")
            else:
                st.caption(f"{diff.upper()}: нет задачи в датасете")

# =========================
# UI
# =========================
st.set_page_config(page_title="Онлайн репетитор", layout="wide")
st.title("📚 Онлайн репетитор по математике")

ensure_files()
tasks_df, TOPIC_COL, TEXT_COL = load_tasks()
model = load_model()

st.sidebar.header("Настройки")
student_id = st.sidebar.text_input("student_id / ник", value="1").strip()

prog = get_student_progress(student_id) if student_id else {"xp": 0, "level": 1, "streak": 0}
st.sidebar.metric("XP", prog["xp"])
st.sidebar.metric("Уровень", prog["level"])
st.sidebar.caption("XP сохраняется в progress.json")

mode = st.sidebar.radio(
    "Режим",
    ["🎯 Задача → тема", "🧠 Рекомендации ученику", "👨‍🏫 Учитель (аналитика + XP)"]
)

# -------------------------
# 1) Задача -> тема
# -------------------------
if mode == "🎯 Задача → тема":
    user_text = st.text_area("Введи текст задачи:", height=140)

    if st.button("🔍 Определить тему"):
        if not user_text.strip():
            st.warning("Введи задачу.")
        else:
            predicted_topic, h1, h2 = get_hints_for_text(user_text, model, topic_blocks)

            st.success(f"Тема (модель): **{predicted_topic}**")

            with st.expander("💡 Подсказка 1", expanded=False):
                st.write(h1)
            with st.expander("💡 Подсказка 2", expanded=False):
                st.write(h2)

            if predicted_topic in topic_blocks:
                show_topic_block(predicted_topic)
            else:
                st.warning("Для этой темы нет topic_blocks. Добавь её в topic_blocks.py")

            st.subheader("📝 Тренировка (1 задача на уровень)")
            cols = st.columns(3)
            for col, diff in zip(cols, ["easy", "medium", "hard"]):
                with col:
                    st.write(f"**{diff.upper()}**")
                    t = pick_task(tasks_df, TOPIC_COL, TEXT_COL, predicted_topic, diff)
                    st.write(t if t else "Нет задач этого уровня.")

# -------------------------
# 2) Рекомендации ученику
# -------------------------
elif mode == "🧠 Рекомендации ученику":
    st.subheader("🧠 Рекомендации по слабым темам (из attempts_log.csv)")
    if not student_id:
        st.warning("Введи student_id слева.")
    else:
        recommend_for_student_streamlit(student_id, tasks_df, TOPIC_COL, TEXT_COL)

# -------------------------
# 3) Учитель: аналитика + XP (НЕ прыгает задача)
# -------------------------
else:
    st.subheader("👨‍🏫 Аналитика + засчитывание задач")
    st.caption("Этот режим заполняет attempts_log.csv и начисляет XP в progress.json")

    st.write("**Распределение задач по темам**")
    st.bar_chart(tasks_df[TOPIC_COL].value_counts())

    st.markdown("---")
    st.write("**Засчитать попытку**")

    topic_for_log = st.selectbox("Тема", sorted(tasks_df[TOPIC_COL].unique()), key="teacher_topic")
    diff_for_log = st.selectbox("Сложность", ["easy", "medium", "hard"], index=1, key="teacher_diff")

    task_key = f"teacher_task_{topic_for_log}_{diff_for_log}"

    if task_key not in st.session_state:
        st.session_state[task_key] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_for_log, diff_for_log)

    task_text = st.session_state[task_key]

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🔄 Следующая задача"):
            st.session_state[task_key] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_for_log, diff_for_log)
            st.rerun()

    if task_text:
        st.write("**Задача:**")
        st.write(task_text)

        correct = st.radio("Решено правильно?", ["Да", "Нет"], horizontal=True, key="teacher_correct")

        if st.button("✅ Засчитать и дать XP"):
            if not student_id:
                st.warning("Введи student_id слева.")
            else:
                log_attempt(student_id, topic_for_log, diff_for_log, 1 if correct == "Да" else 0, task_text)
                updated, reward = add_xp(student_id, diff_for_log)
                st.success(f"Записано. +{reward} XP. Уровень: {updated['level']}, XP: {updated['xp']}")

                st.session_state[task_key] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_for_log, diff_for_log)
                st.rerun()
    else:
        st.warning("Нет задач для этой темы/сложности.")
