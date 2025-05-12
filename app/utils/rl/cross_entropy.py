import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


def build_probs_matrix(transitions_df: pd.DataFrame) -> pd.DataFrame:
    transition_counts = transitions_df.groupby(["from", "to"]).size().reset_index(name="count")
    transition_counts["prob"] = transition_counts["count"] / transition_counts.groupby("from")["count"].transform("sum")
    return transition_counts.pivot(index="from", columns="to", values="prob").fillna(0)


def simulate_paths(
    probs_matrix: pd.DataFrame,
    transitions_df: pd.DataFrame,
    terminal_states: List[str],
    start_state: str,
    reward_map: Dict[str, float],
    n: int = 100
) -> List[Tuple[List[str], float]]:
    paths = []
    for _ in range(n):
        state = start_state
        path = [state]
        total_reward = 0
        for _ in range(50):
            if state not in probs_matrix.index or probs_matrix.loc[state].sum() == 0:
                break
            probs = probs_matrix.loc[state]
            next_state = np.random.choice(probs.index, p=probs / probs.sum())
            time = transitions_df[(transitions_df["from"] == state) & (transitions_df["to"] == next_state)]["time"].mean()
            reward = -time + reward_map.get(next_state, 0)
            path.append(next_state)
            total_reward += reward
            if next_state in terminal_states:
                break
            state = next_state
        paths.append((path, total_reward))
    return paths


def train_cross_entropy(
    transitions_df: pd.DataFrame,
    terminal_states: List[str],
    start_state: str,
    reward_map: Dict[str, float],
    iterations: int = 20,
    n_samples: int = 200,
    elite_frac: float = 0.2
) -> pd.DataFrame:
    probs_matrix = build_probs_matrix(transitions_df)

    for _ in range(iterations):
        samples = simulate_paths(probs_matrix, transitions_df, terminal_states, start_state, reward_map, n_samples)
        elite = sorted(samples, key=lambda x: -x[1])[:int(n_samples * elite_frac)]

        updated = defaultdict(lambda: defaultdict(int))
        for path, _ in elite:
            for i in range(len(path) - 1):
                updated[path[i]][path[i + 1]] += 1

        flat_data = [
            {"from": f, "to": t, "prob": count / sum(updated[f].values())}
            for f in updated for t, count in updated[f].items()
        ]
        probs_matrix = pd.DataFrame(flat_data).pivot(index="from", columns="to", values="prob").fillna(0)

    return probs_matrix