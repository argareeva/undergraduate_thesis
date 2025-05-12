import os
import streamlit as st
import pandas as pd
from sberpm.imitation import Simulation

from utils.preprocessing import validate_columns, detect_start_state, preprocess_data_with_new_stage
from utils.process_mining import render_graph, show_graph, generate_and_prepare_holder, analyze_and_show
from utils.metrics import process_mining_metrics, initial_metrics, summarize_paths_metrics, render_strategy_results, evaluate_success_rate, get_best_ppo, get_best_td3

from utils.rl.q_learning import QLearningSimulator, simulate_q_policy
from utils.rl.cross_entropy import train_cross_entropy, simulate_paths
from utils.rl.genetic import run_genetic_algorithm
from utils.rl.advanced import ppo, td3, env

st.set_page_config(page_title="Process Mining + RL", layout="wide")
st.title("Аналитический инструмент по процессу найма")

# Выбор режима работы: PM или RL
mode = st.sidebar.selectbox("Выберите режим", ["Process Mining (PM)", "Reinforcement Learning (RL)"])

# ========== PM MODE ==========
if mode == "Process Mining (PM)":
    st.header("Анализ процесса")

    file = st.file_uploader("Для анализа процесса загрузите CSV-файл (с разделителем ‘;’). Убедитесь, что файл содержит колонки: case_id, line_id, from, to, time", type=["csv"], key="pm_upload")
    if file:
        os.makedirs("data", exist_ok=True)
        file_path = "data/pm_data.csv"
        with open(file_path, "wb") as f:
            f.write(file.read())

        df = pd.read_csv(file_path, sep=";")
        validate_columns(df)

        holder = preprocess_data_with_new_stage(file_path)

        # Визуализация графа
        st.subheader("Граф бизнес-процесса")
        graph_path = render_graph(holder, filename="real_graph", format="png")
        st.image(graph_path, caption="Граф реального бизнес-процесса")

        st.markdown("### Метрики")
        metrics = process_mining_metrics(holder)
        st.subheader("Метрики активности")
        st.dataframe(metrics["activity_metric"])
        st.subheader("Метрики переходов")
        st.dataframe(metrics["transition_metric"])

        st.subheader("Настройка успешного терминального состояний")
        success_state = st.selectbox("Выберите успешное терминальное состояние:", df["to"].unique(), key="success_state")

        st.subheader("Исходные метрики на реальных данных")
        metrics = initial_metrics(holder, success_state)
        st.metric("Средняя длительность кейса", f"{metrics['avg_case_duration']:.2f} ч.")
        st.metric("Средняя длительность успешного кейса", f"{metrics['real_success_avg']:.2f} ч.")

        st.subheader("Ручная оптимизация процесса")
        speed_up_stages = st.multiselect("Стадии для ускорения (scale=0.5):", holder.data["to"].unique(), key="pm_speedup")
        delete_nodes = st.multiselect("Удалить узлы:", holder.data["to"].unique(), key="pm_delete_nodes")
        delete_loops = st.multiselect("Удалить циклы:", holder.data["to"].unique(), key="pm_delete_loops")
        edges_text = st.text_area("Удалить переходы (формат: From -> To, по одному на строку):", key="pm_edges")

        if st.button("Применить оптимизацию и запустить симуляцию", key="pm_optimize"):
            sim = Simulation(holder)
            for s in speed_up_stages:
                sim.scale_time_node(s, scale=0.5)
            for s in delete_nodes:
                sim.delete_node(s)
            for s in delete_loops:
                sim.delete_loop(s)
            for line in edges_text.splitlines():
                if "->" in line:
                    f, t = map(str.strip, line.split("->"))

                    missing_from_states = [f] if f not in df["from"].unique() else []
                    missing_to_states = [t] if t not in df["to"].unique() else []

                    if missing_from_states and missing_to_states:
            	        st.error(f"Ошибка: Начальные состояния отсутствуют в данных: {', '.join(missing_from_states)}. "
                     f"Конечные состояния отсутствуют в данных: {', '.join(missing_to_states)}.")
                    elif missing_from_states:
            	        st.error(f"Ошибка: Начальные состояния отсутствуют в данных: {', '.join(missing_from_states)}.")
            	        st.stop()
                    elif missing_to_states:
            	        st.error(f"Ошибка: Конечные состояния отсутствуют в данных: {', '.join(missing_to_states)}.")
            	        st.stop()

                    sim.delete_edge(f, t)

            # Результаты оптимизации
            st.subheader("Результаты оптимизации")
            data_opt, opt_holder = generate_and_prepare_holder(sim, iterations=1000)
            opt_results = analyze_and_show(opt_holder, success_state, title="optimized_graph", label="Оптимизированный")
            st.image(opt_results["graph_path"], caption="Оптимизированный граф бизнес-процесса")
            st.metric("Средняя длительность кейса", f"{opt_results['avg_case_duration']:.2f} ч.")
            st.metric("Средняя длительность успешного кейса", f"{opt_results['real_success_avg']:.2f} ч.")

            st.subheader("Вывод")
            st.write("Ручная оптимизация позволяет уменьшить время завершения процесса за счет ускорения стадий, удаления ненужных переходов, циклов и узлов. Однако данный метод дает только числовые результаты и визуализацию. Чтобы определить стратегию выполнения процесса, перейдите в раздел Reinforcement Learning.")

# ========== RL MODE ==========
if mode == "Reinforcement Learning (RL)":
    st.header("Тестирование RL-алгоритмов")

    file = st.file_uploader("Для анализа процесса загрузите CSV-файл (с разделителем ‘;’). Убедитесь, что файл содержит колонки: case_id, line_id, from, to, time", type=["csv"], key="rl_upload")
    if file:
        with open("data/rl_data.csv", "wb") as f:
            f.write(file.getbuffer())
        df = pd.read_csv("data/rl_data.csv", sep=";")
        validate_columns(df)

        st.subheader("Настройка терминальных состояний")
        success_state = st.selectbox("Выберите успешное терминальное состояние:", df["to"].unique(), key="success_state_rl")

        other_terminal_states = st.multiselect("Выберите остальные терминальные состояния:", df["to"].unique(), key="other_terminals")
        terminal_states = [success_state] + other_terminal_states

        st.subheader("Настройка оценки состояний")
        reward_map = {}
        st.write("Укажите оценку для каждого состояния:")
        for stage in df["to"].unique():
            reward_map[stage] = st.number_input(f"Оценка состояния '{stage}':", value=0, step=50, key=f"reward_{stage}")

        alg = st.selectbox("Выберите алгоритм", ["Q-learning", "Cross Entropy", "Genetic Algorithm", "PPO", "TD3"], key="rl_alg")
        
        st.markdown("---")
        st.subheader("Тестирование гипотез")
        edges_text = st.text_area("Удалить переходы (формат: From -> To, по одному на строку):", key="rl_edges")
        exclude_nodes = st.multiselect("Исключить узлы:", df["from"].unique(), key="rl_exclude_nodes")

        df_mod = df.copy()
        for line in edges_text.splitlines():
            if "->" in line:
                f, t = map(str.strip, line.split("->"))

                missing_from_states = [f] if f not in df["from"].unique() else []
                missing_to_states = [t] if t not in df["to"].unique() else []

                if missing_from_states and missing_to_states:
            	    st.error(f"Ошибка: Начальные состояния отсутствуют в данных: {', '.join(missing_from_states)}. "
                     f"Конечные состояния отсутствуют в данных: {', '.join(missing_to_states)}.")
                elif missing_from_states:
            	    st.error(f"Ошибка: Начальные состояния отсутствуют в данных: {', '.join(missing_from_states)}.")
            	    st.stop()
                elif missing_to_states:
            	    st.error(f"Ошибка: Конечные состояния отсутствуют в данных: {', '.join(missing_to_states)}.")
            	    st.stop()

                df_mod = df_mod[~((df_mod["from"] == f) & (df_mod["to"] == t))]
        for node in exclude_nodes:
            df_mod = df_mod[(df_mod["from"] != node) & (df_mod["to"] != node)]

        transition_times = df_mod.groupby(["from", "to"])["time"].mean().to_dict()

        start_state = detect_start_state(df)

        if st.button("Запустить алгоритм"):
            if alg == "Q-learning":
                q_agent = QLearningSimulator(df_mod, reward_map, terminal_states, start_state)
                q_table = q_agent.train_q_learning()
                q_paths = simulate_q_policy(q_table, df_mod, terminal_states, start_state)
                st.markdown(render_strategy_results("Q-learning", q_paths, success_state, transition_times=transition_times))

                st.subheader("Вывод")
                st.markdown("Q-learning обучается на данных переходов между состояниями, определяя оптимальные действия на каждом этапе. Алгоритм обновляет значения в Q-таблице на основе накопленного вознаграждения, стремясь минимизировать суммарные затраты на достижение целевого состояния.\n\n" +
                "*Если вы используете тестовый датасет, для генерации оптимальной стратегии с этим алгоритмом необходимо:*\n\n" +
                "- *Удалить переходы: Интервью -> Интервью, Рассмотрение рекрутером -> Не выходит на связь, Рассмотрение рекрутером -> Самоотказ, Рассмотрение рекрутером -> Отклонен, Рассмотрение рекрутером -> Планирование интервью, Планирование интервью -> Отклонен, Рассмотрение заказчиком -> Отклонен, Заполнение анкеты -> Заполнение анкеты;*\n\n" + 
                "- *Исключить узлы: Оценка кандидата, Готов к переводу.*")

            elif alg == "Cross Entropy":
                cem_matrix = train_cross_entropy(df_mod, terminal_states, start_state, reward_map)
                cem_paths = simulate_paths(cem_matrix, df_mod, terminal_states, start_state, reward_map)
                st.markdown(render_strategy_results("Cross Entropy", cem_paths, success_state, transition_times=transition_times))

                st.subheader("Вывод")
                st.markdown("Cross Entropy обучается на множестве случайных траекторий, выделяя наиболее успешные и обновляя параметры для генерации лучших решений. Алгоритм стремится минимизировать время и количество шагов до целевого состояния.\n\n" +
                "*Если вы используете тестовый датасет, для генерации оптимальной стратегии с этим алгоритмом необходимо:*\n\n" +
                "- *Удалить переходы: Интервью -> Интервью, Рассмотрение рекрутером -> Не выходит на связь, Рассмотрение рекрутером -> Самоотказ, Рассмотрение рекрутером -> Отклонен;*\n\n" + 
                "- *Исключить узлы: Оценка кандидата, Готов к переводу.*")

            elif alg == "Genetic Algorithm":
                ga_paths = run_genetic_algorithm(df_mod, terminal_states, transition_times, start_state, reward_map)
                st.markdown(render_strategy_results("Genetic Algorithm", ga_paths, success_state, transition_times=transition_times))

                st.subheader("Вывод")
                st.markdown("Генетический алгоритм ищет оптимальные пути, комбинируя и мутируя возможные решения. Алгоритм выбирает наиболее удачные стратегии и адаптирует их, чтобы минимизировать общее время достижения целевого состояния.\n\n" +
                "*Если вы используете тестовый датасет, для генерации оптимальной стратегии с этим алгоритмом необходимо:*\n\n" +
                "- *Удалить переходы: Интервью -> Интервью, Рассмотрение рекрутером -> Не выходит на связь, Рассмотрение рекрутером -> Самоотказ, Рассмотрение рекрутером -> Отклонен, Рассмотрение рекрутером -> Планирование интервью, Планирование интервью -> Отклонен, Рассмотрение заказчиком -> Отклонен;*\n\n" + 
                "- *Исключить узлы: Оценка кандидата, Готов к переводу.*")

            elif alg == "PPO":
                env = env.MaskedProcessEnv(df_mod, start_state, terminal_states, reward_map)
                model = ppo.train_custom_ppo(env, reward_map)
                results = ppo.simulate_masked_ppo(model, env)

                best_path, duration = get_best_ppo(results, success_state, transition_times=transition_times)
                success_rate = evaluate_success_rate(model, env, success_state, n_episodes=100)

                if best_path:
                    st.write("Лучшая стратегия:", " → ".join(best_path))
                    st.write("Длительность:", duration)
                    st.write(f"Успешность: {success_rate}%")
                else:
                    st.write("Успешных стратегий не найдено.")

                st.subheader("Вывод")
                st.markdown("Алгоритм PPO оптимизирует стратегию через градиентное обучение, обновляя политику на основе текущих состояний и полученных вознаграждений. Он балансирует между исследованием новых действий и эксплуатацией уже найденных оптимальных стратегий.\n\n" +
                "*Если вы используете тестовый датасет, для генерации оптимальной стратегии с этим алгоритмом необходимо:*\n\n" +
                "- *Удалить переходы: Интервью -> Интервью, Заполнение анкеты -> Заполнение анкеты.*")

            elif alg == "TD3":
                env = env.MaskedProcessEnv(df_mod, start_state, terminal_states, reward_map)
                model = td3.train_td3(env, transition_times, reward_map)
                results = td3.simulate_td3(model, env, transition_times=transition_times)
                
                best_path, duration, success_rate = get_best_td3(results, success_state, transition_times=transition_times)

                if best_path:
                    st.write("Лучшая стратегия:", " → ".join(best_path))
                    st.write("Длительность:", duration)
                    st.write("Успешность:", success_rate)
                else:
                    st.write("Успешных стратегий не найдено.")

                st.subheader("Вывод")
                st.markdown("Алгоритм TD3 улучшает стратегию с помощью градиентного спуска, используя два критика для уменьшения переоценки действия. Он минимизирует ошибку в оценке вознаграждений, стабилизируя обучение за счет задержки обновления политики.\n\n" +
                "*Если вы используете тестовый датасет, для генерации оптимальной стратегии с этим алгоритмом необходимо:*\n\n" +
                "- *Удалить переходы: Интервью -> Интервью, Заполнение анкеты -> Заполнение анкеты.*")