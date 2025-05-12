from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import numpy as np
from sberpm import DataHolder
from sberpm.metrics import ActivityMetric, TransitionMetric
import torch


def process_mining_metrics(data_holder):
    
    # Метрика активностей
    activity_metric = ActivityMetric(data_holder, time_unit='h').apply().reset_index()
    activity_cols = ['index', 'count', 'unique_ids_num', 'aver_count_in_trace', 'loop_percent', 'throughput']

    # Метрика переходов
    transition_metric = TransitionMetric(data_holder, time_unit='h').apply().reset_index()
    transition_cols = ['index', 'count', 'unique_ids_num', 'aver_count_in_trace', 'loop_percent', 'throughput']

    return {
    "activity_metric": activity_metric[activity_cols],
    "transition_metric": transition_metric[transition_cols]
    }


def initial_metrics(data_holder, success_state=None):
    avg_duration = data_holder.data.groupby('case_id')["time"].sum().mean()

    successful_case_ids = data_holder.data[data_holder.data["to"] == success_state]["case_id"].unique()
    successful_cases_duration = data_holder.data[data_holder.data["case_id"].isin(successful_case_ids)]
    real_success_avg = successful_cases_duration.groupby("case_id")["time"].sum().mean()

    return {
        "avg_case_duration": avg_duration,
        "real_success_avg": real_success_avg
    }


def calculate_avg_time_successful(
    paths: List[Tuple[List[str], float]],
    success_state: str,
    transition_times: Optional[Dict[Tuple[str, str], float]] = None,
) -> Optional[float]:
    if transition_times:
        success_paths = [p for p, _ in paths if p[-1] == success_state]
        times = [
            sum(transition_times.get((p[i], p[i + 1]), 0) for i in range(len(p) - 1))
            for p in success_paths
        ]
        return round(np.mean(times), 2) if times else None
    else:
        success_scores = [abs(t) for p, t in paths if p[-1] == success_state]
        return round(np.mean(success_scores), 2) if success_scores else None


def summarize_paths_metrics(
    paths: List[Tuple[List[str], float]],
    success_state: str,
    transition_times: Optional[Dict[Tuple[str, str], float]] = None
) -> Tuple[float, float, float, Optional[float]]:
    total = len(paths)
    successes = [p for p, _ in paths if p[-1] == success_state]

    success_rate = 100 * len(successes) / total if total > 0 else 0
    avg_length = sum(len(p) for p, _ in paths) / total if total > 0 else 0

    if transition_times:
        all_times = [
            sum(transition_times.get((p[i], p[i + 1]), 0) for i in range(len(p) - 1))
            for p, _ in paths
        ]
        avg_time = round(np.mean(all_times), 2) if all_times else 0.0
    else:
        avg_time = sum(abs(t) for _, t in paths) / total if total > 0 else 0

    avg_success_time = calculate_avg_time_successful(paths, success_state, transition_times)
    return success_rate, avg_length, avg_time, avg_success_time


def extract_strategy_from_paths(paths: List[Tuple[List[str], float]]) -> List[str]:
    counter = defaultdict(lambda: defaultdict(int))
    for path, _ in paths:
        for i in range(len(path) - 1):
            counter[path[i]][path[i + 1]] += 1

    strategy_lines = []
    for state in counter:
        if counter[state]:
            best = max(counter[state], key=counter[state].get)
            strategy_lines.append(f"{state} -> {best}")
    return strategy_lines


def render_strategy_results(
    title: str,
    paths: List[Tuple[List[str], float]],
    success_state: str,
    transition_times: Optional[Dict[Tuple[str, str], float]] = None
) -> str:
    success_rate, avg_length, avg_time, avg_success_time = summarize_paths_metrics(
        paths, success_state, transition_times
    )
    strategy = extract_strategy_from_paths(paths)

    output = f"### {title}\n"
    output += f"**Успешность:** {success_rate:.1f}%  \n"
    output += f"**Средняя длина пути:** {avg_length:.2f}  \n"
    output += f"**Среднее время всех кейсов:** {avg_time:.2f} ч  \n"
    output += (
        f"**Среднее время успешных кейсов:** {avg_success_time:.2f} ч\n"
        if avg_success_time is not None
        else "**Нет успешных кейсов**\n"
    )

    output += "\n**Стратегия:**\n"
    output += "\n".join([f"- {line}" for line in strategy])

    return output


def evaluate_success_rate(model, env, success_state, n_episodes=100):
    success_count = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        for _ in range(50):
            obs_vec = np.concatenate([[obs["state"]], obs["action_mask"]])
            obs_tensor = torch.tensor([obs_vec], dtype=torch.float32)
            mask_tensor = torch.tensor([obs["action_mask"]], dtype=torch.float32)
            dist, _ = model(obs_tensor.unsqueeze(0), mask_tensor.unsqueeze(0))
            action = int(dist.sample().item())
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if env.idx_to_state[obs['state']] == success_state:
            success_count += 1
    return round(success_count / n_episodes * 100, 2)


def get_best_ppo(results, success_state, transition_times):
    successful = [(path, reward) for path, reward in results if path[-1] == success_state]
    if not successful:
        return None, None

    best_path, _ = max(successful, key=lambda x: x[1])
    total_time = sum(transition_times.get((best_path[i], best_path[i+1]), 0) for i in range(len(best_path) - 1))

    return best_path, round(total_time, 2)


def get_best_td3(results, success_state, transition_times):
    successful = [(p, r) for p, r in results if p[-1] == success_state]
    if not successful:
        return None, None, 0.0
    best_path, _ = max(successful, key=lambda x: x[1])
    duration = sum(transition_times.get((best_path[i], best_path[i+1]), 0) for i in range(len(best_path)-1))
    success_rate = round(len(successful) / len(results) * 100, 2)
    return best_path, round(duration, 2), success_rate
