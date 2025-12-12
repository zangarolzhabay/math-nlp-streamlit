# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import random

import importlib
import topic_blocks as tb
importlib.reload(tb)

topic_blocks = tb.topic_blocks


# =========================
# ФАЙЛЫ (лежать рядом в папке проекта)
# =========================
DATA_PATH = Path("math_tasks.csv")        # датасет задач
MODEL_PATH = Path("nlp_model (1).pkl")        # твоя NLP модель (joblib/pkl)
PIVOT_PATH = Path("pivot_table.csv")      
PRACTICE_LOG_PATH = Path("practice_log.csv")  # пустой лог для режима 3

# =========================
# XP / уровни (режим 3)
# =========================
xp_rewards = {"easy": 5, "medium": 10, "hard": 20}
level_thresholds = {1: 0, 2: 50, 3: 120, 4: 250, 5: 250, 6: 500, 7: 1000}

# =========================
# УТИЛИТЫ
# =========================
def clean_text(x: str) -> str:
    return " ".join(str(x).lower().strip().split())

def ensure_practice_log():
    if not PRACTICE_LOG_PATH.exists():
        pd.DataFrame(columns=[
            "ts", "student_id", "topic", "difficulty", "task_text", "xp_awarded"
        ]).to_csv(PRACTICE_LOG_PATH, index=False, encoding="utf-8-sig")

def log_practice(student_id: str, topic: str, difficulty: str, task_text: str, xp_awarded: int):
    ensure_practice_log()
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "student_id": str(student_id),
        "topic": str(topic),
        "difficulty": str(difficulty).lower(),
        "task_text": str(task_text),
        "xp_awarded": int(xp_awarded),
    }
    df = pd.read_csv(PRACTICE_LOG_PATH, encoding="utf-8-sig")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(PRACTICE_LOG_PATH, index=False, encoding="utf-8-sig")

def get_progress_from_practice_log(student_id: str):
    ensure_practice_log()
    df = pd.read_csv(PRACTICE_LOG_PATH, encoding="utf-8-sig")
    if df.empty:
        xp = 0
    else:
        df["student_id"] = df["student_id"].astype(str)
        xp = int(df[df["student_id"] == str(student_id)]["xp_awarded"].sum())

    lvl = 1
    for k, v in sorted(level_thresholds.items()):
        if xp >= v:
            lvl = k
    return {"xp": xp, "level": lvl}

def pick_task(df: pd.DataFrame, topic_col: str, text_col: str, topic: str, difficulty: str):
    sub = df[df[topic_col].astype(str) == str(topic)].copy()
    if sub.empty:
        return None

    if "difficulty" in sub.columns:
        sub["difficulty"] = sub["difficulty"].astype(str).str.lower()
        dsub = sub[sub["difficulty"] == str(difficulty).lower()]
        if not dsub.empty:
            sub = dsub

    # важное: без random_state, иначе будет часто повторяться одинаково
    return sub.sample(1)[text_col].values[0]

def show_topic_block(topic_key: str):
    info = topic_blocks.get(topic_key)
    if not info:
        st.warning("Для этой темы нет topic_blocks (добавь в topic_blocks.py).")
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

@st.cache_data
def load_tasks():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if "topic_clean" in df.columns:
        topic_col = "topic_clean"
    elif "topic" in df.columns:
        topic_col = "topic"
    else:
        raise ValueError("В math_tasks.csv нет колонки topic_clean или topic")

    text_col = "task_text" if "task_text" in df.columns else df.columns[0]

    # если нет difficulty — создаём, чтобы режим 3 работал
    if "difficulty" not in df.columns:
        df["difficulty"] = "medium"

    # чистим типы
    df[topic_col] = df[topic_col].astype(str)
    df[text_col] = df[text_col].astype(str)
    df["difficulty"] = df["difficulty"].astype(str).str.lower()

    return df, topic_col, text_col

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_pivot():
    if not PIVOT_PATH.exists():
        return None
    pivot = pd.read_csv(PIVOT_PATH, encoding="utf-8-sig")

    # ожидаем: 1-я колонка student_id или индекс
    # делаем student_id индексом
    if "student_id" in pivot.columns:
        pivot["student_id"] = pivot["student_id"].astype(str)
        pivot = pivot.set_index("student_id")
    else:
        # если нет student_id, считаем что первая колонка — это student_id
        pivot.iloc[:, 0] = pivot.iloc[:, 0].astype(str)
        pivot = pivot.set_index(pivot.columns[0])

    # остальные колонки должны быть темами с числами 0..1
    for c in pivot.columns:
        pivot[c] = pd.to_numeric(pivot[c], errors="coerce")
    return pivot

STUDENTS = {
    "1": "Айымжан",
    "2": "Али",
    "3": "Нурали",
    "4": "Нурайым",
    "5": "Айару",
    "6": "Аружан",
    "7": "Данияр",
    "8": "Айсана",
    "9": "Темирлан",
    "10": "Жанерке",
    "11": "Ерасыл",
    "12": "Амина",
    "13": "Арсен",
    "14": "Мадина",
    "15": "Бекзат",
    "16": "Айбек",
    "17": "Салтанат",
    "18": "Нурислам",
    "19": "Диас",
    "20": "Камила",
    "21": "Рамазан",
    "22": "Алина",
    "23": "Мейиржан",
    "24": "Айдана",
    "25": "Самат",
    "26": "Жасмин",
    "27": "Ильяс",
    "28": "Карина",
    "29": "Санжар",
    "30": "Малика",
}

NAME_TO_ID = {name: sid for sid, name in STUDENTS.items()}

# =========================
# UI
# =========================
st.set_page_config(page_title="Онлайн репетитор", layout="wide")
st.title("Онлайн репетитор по математике")

# загрузки
tasks_df, TOPIC_COL, TEXT_COL = load_tasks()
model = load_model()
pivot_table = load_pivot()
ensure_practice_log()

# Sidebar
st.sidebar.header("Настройки")
student_name = st.sidebar.selectbox("Ученик", list(STUDENTS.values()), index=0)
student_id = NAME_TO_ID[student_name]   

prog = get_progress_from_practice_log(student_id) if student_id else {"xp": 0, "level": 1}
st.sidebar.metric("XP", prog["xp"])
st.sidebar.metric("Уровень", prog["level"])
st.sidebar.caption("XP считается из practice_log.csv (только режим 3)")

mode = st.sidebar.radio("Режим", [
    "1) NLP: задача → тема",
    "2) Рекомендации по pivot_table",
    "3) Практика + XP "
])

# =========================
# 1) NLP: задача → тема
# =========================
if mode == "1) NLP: задача → тема":
    st.subheader("1) Введи задачу, модель скажет тему")

    user_text = st.text_area("Текст задачи:", height=160)

    if st.button("🔍 Определить тему"):
        if not user_text.strip():
            st.warning("Введи текст задачи.")
        else:
            pred = model.predict([clean_text(user_text)])[0]
            st.success(f"Тема (модель): **{pred}**")

            # теория (из твоего topic_blocks)
            show_topic_block(pred)

            # мини-тренировка: 1 задача на уровень
            st.markdown("---")
            st.subheader("📝 Тренировка по этой теме (1 задача на уровень)")
            cols = st.columns(3)
            for col, diff in zip(cols, ["easy", "medium", "hard"]):
                with col:
                    st.write(f"**{diff.upper()}**")
                    t = pick_task(tasks_df, TOPIC_COL, TEXT_COL, pred, diff)
                    st.write(t if t else "Нет задач этого уровня в датасете.")

# =========================
# 2) Рекомендации по pivot_table.csv 
# =========================
elif mode == "2) Рекомендации по pivot_table":
    st.subheader(f" Рекомендации для ученика: {student_name}")
    st.caption("Этот режим ничего не записывает и не сохраняет. Только читает pivot_table.csv.")

    if pivot_table is None:
        st.error("Файл pivot_table.csv не найден рядом со Streamlit. Положи его в папку проекта.")
    elif not student_id:
        st.warning("Введи student_id слева.")
    elif str(student_id) not in pivot_table.index:
        st.warning("Этого student_id нет в pivot_table.csv.")
    else:
        row = pivot_table.loc[str(student_id)].dropna()

        if row.empty:
            st.info("По ученику нет значений (пустая строка в pivot_table).")
        else:
            weak_topics = row[row < 0.5].sort_values().index.tolist()
            st.write(f"❌ Слабые темы : **{weak_topics if weak_topics else 'нет'}**")

            for topic in weak_topics:
                st.markdown("---")
                st.subheader(f"📌 Тема: {topic}")

                # теория
                show_topic_block(topic)

                # практика: 
                st.write("🧠 Практика:")

                tasks_order = [
                    ("Задача 1", "easy"),
                    ("Задача 2", "medium"),
                    ("Задача 3", "hard")
                ]

                for label, diff in tasks_order:
                    t = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic, diff)
                    if t:
                        st.markdown(f"**{label}**")
                        st.write(t)
                    else:
                        st.caption(f"{label}: нет задачи")
# =========================
# 3) Практика + XP (клик = +XP)
# =========================
else:
    st.subheader("3) Практика + XP")
    st.caption("Нажал «Следующая задача» = получил XP. Лог пишется в practice_log.csv.")

    if not student_id:
        st.warning("Введи student_id слева.")
        st.stop()

    # выбор темы и сложности
    all_topics = sorted(tasks_df[TOPIC_COL].unique())
    topic_choice = st.selectbox("Тема", all_topics, key="practice_topic")
    diff_choice = st.selectbox("Сложность", ["easy", "medium", "hard"], index=1, key="practice_diff")

    # фиксируем задачу в session_state, чтобы она НЕ менялась от кликов/радио
    key_task = f"current_task_{student_id}"

    if key_task not in st.session_state:
        st.session_state[key_task] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_choice, diff_choice)

    # если пользователь поменял тему/сложность, сбрасываем задачу
    # (иначе будет показывать старую)
    key_last = f"last_params_{student_id}"
    last = st.session_state.get(key_last)
    cur_params = (topic_choice, diff_choice)
    if last != cur_params:
        st.session_state[key_last] = cur_params
        st.session_state[key_task] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_choice, diff_choice)

    st.markdown("---")
    st.write("**Текущая задача:**")
    st.write(st.session_state[key_task] if st.session_state[key_task] else "Нет задач для этого выбора.")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("➡️ Следующая задача (+XP)", use_container_width=True):
            current_task = st.session_state.get(key_task)

            # начисляем XP даже если ученик не решал (как ты хотел)
            reward = xp_rewards.get(diff_choice.lower(), 5)

            if current_task:
                log_practice(student_id, topic_choice, diff_choice, current_task, reward)

            prog2 = get_progress_from_practice_log(student_id)
            st.success(f"+{reward} XP. Уровень: {prog2['level']} | XP: {prog2['xp']}")

            # выдаём новую задачу
            st.session_state[key_task] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_choice, diff_choice)

    with colB:
        if st.button("🔄 Сбросить текущую задачу", use_container_width=True):
            st.session_state[key_task] = pick_task(tasks_df, TOPIC_COL, TEXT_COL, topic_choice, diff_choice)

    st.markdown("---")
    st.write("**Твой прогресс (из practice_log.csv):**")
    prog3 = get_progress_from_practice_log(student_id)
    st.metric("XP", prog3["xp"])
    st.metric("Уровень", prog3["level"])
