# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import random

from topic_blocks import topic_blocks  # твой готовый словарь

# =========================
# Пути
# =========================
DATA_PATH = Path("math_tasks.csv")
MODEL_PATH = Path("nlp_model (1).pkl")      # переименуй файл так (без (1))
PROGRESS_PATH = Path("progress.json")
PIVOT_PATH = Path("pivot_table.csv")   # экспортируешь из Colab

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

def ensure_progress_file():
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("{}", encoding="utf-8")

def load_progress() -> dict:
    ensure_progress_file()
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

def get_hints_for_text(text: str, model):
    cleaned = clean_math_text(text)
    predicted_topic = model.predict([cleaned])[0]

    info = topic_blocks.get(predicted_topic, {})
    hint1 = info.get("hint1") or "Подумай, какое правило/формула подходит."
    hint2 = info.get("hint2") or "Разбей на шаги: дано → найти → формула → подстановка."

    return predicted_topic, hint1, hint2

# =========================
# Загрузка данных/модели/pivot
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

    df[topic_col] = df[topic_col].astype(str)
    df["difficulty"] = df["difficulty"].astype(str).str.lower()

    return df, topic_col, text_col

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_pivot():
    # pivot_table.csv должен иметь: student_id + колонки тем (значения 0..1)
    pivot = pd.read_csv(PIVOT_PATH)
    pivot["student_id"] = pivot["student_id"].astype(str)
    pivot = pivot.set_index("student_id")
    return pivot

# =========================
# UI
# =========================
st.set_page_config(page_title="Онлайн репетитор по математике", layout="wide")
st.title("Онлайн репетитор по математике: тема → теория → практика")

tasks_df, TOPIC_COL, TEXT_COL = load_tasks()
model = load_model()
pivot_table = load_pivot()

st.sidebar.header("Настройки")
student_id = st.sidebar.text_input("student_id / ник", value="1").strip()

ensure_progress_file()
prog = get_student_progress(student_id) if student_id else {"xp": 0, "level": 1, "streak": 0}
st.sidebar.metric("XP", prog["xp"])
st.sidebar.metric("Уровень", prog["level"])
st.sidebar.caption("XP сохраняется в progress.json")

mode = st.sidebar.radio("Режим", ["🎯 Задача → тема", "🧠 Рекомендации ученику"])

# =========================
# Режим 1: задача → тема
# =========================
if mode == "🎯 Задача → тема":
    user_text = st.text_area("Введи текст задачи:", height=140)

    if st.button("🔍 Определить тему"):
        if not user_text.strip():
            st.warning("Введи задачу.")
        else:
            predicted_topic, h1, h2 = get_hints_for_text(user_text, model)

            st.success(f"Тема (модель): **{predicted_topic}**")

            # подсказки НЕ сразу (как ты хотел)
            with st.expander("💡 Подсказка 1", expanded=False):
                st.write(h1)
            with st.expander("💡 Подсказка 2", expanded=False):
                st.write(h2)

            if predicted_topic in topic_blocks:
                show_topic_block(predicted_topic)
            else:
                st.warning("Для этой темы нет topic_blocks. Добавь её в topic_blocks.py")

            st.subheader("📝 Тренировка по этой теме (1 задача на уровень)")
            cols = st.columns(3)
            for col, diff in zip(cols, ["easy", "medium", "hard"]):
                with col:
                    t = pick_task(tasks_df, TOPIC_COL, TEXT_COL, predicted_topic, diff)
                    st.write(f"**{diff.upper()}**")
                    st.write(t if t else "Нет задач этого уровня")

            if st.button("✅ Я потренировался (дать XP)"):
                updated, reward = add_xp(student_id, "medium")
                st.success(f"+{reward} XP. Уровень: {updated['level']}, XP: {updated['xp']}")

# =========================
# Режим 2: рекомендации ученику (СРАЗУ из pivot_table)
# =========================
else:
    st.subheader("🧠 Рекомендации ученику")

    if student_id not in pivot_table.index:
        st.warning("⚠️ Нет данных для этого student_id в pivot_table.csv")
        st.stop()

    row = pivot_table.loc[student_id].dropna()
    weak_topics = row[row < 0.5].sort_values().index.tolist()

    st.write(f"❌ Слабые темы (точность < 0.5): **{weak_topics if weak_topics else 'нет'}**")

    for topic in weak_topics:
        st.markdown("---")
        st.markdown(f"### 📌 {topic}")

        if topic in topic_blocks:
            show_topic_block(topic)
        else:
            st.info("Нет topic_blocks для этой темы.")

        st.write("🧠 Практика (1 задача на уровень):")
        for diff in ["easy", "medium", "hard"]:
            t = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic, diff)
            st.write(f"**{diff.upper()}**: {t if t else 'Нет задачи'}")
