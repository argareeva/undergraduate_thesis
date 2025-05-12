import random
from typing import List, Tuple, Dict


def fitness(path: List[str], transition_times: Dict[Tuple[str, str], float], reward_map: Dict[str, float]) -> float:
    score = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        time = transition_times.get(edge)
        if time is None:
            return -1000
        score -= time
    score += reward_map.get(path[-1], 0)
    return score


def run_genetic_algorithm(
    transitions_df,
    terminal_states: List[str],
    transition_times: Dict[Tuple[str, str], float],
    start_state: str,
    reward_map: Dict[str, float],
    population_size: int = 100,
    generations: int = 50
) -> List[Tuple[List[str], float]]:
    transition_dict = transitions_df.groupby("from")["to"].apply(list).to_dict()

    def generate_path():
        path, current = [start_state], start_state
        for _ in range(10):
            if current not in transition_dict:
                break
            next_state = random.choice(transition_dict[current])
            path.append(next_state)
            if next_state in terminal_states:
                break
            current = next_state
        return path

    population = [generate_path() for _ in range(population_size)]

    for _ in range(generations):
        scored = [(p, fitness(p, transition_times, reward_map)) for p in population]
        top = sorted(scored, key=lambda x: -x[1])[:population_size // 2]

        offspring = []
        while len(offspring) < population_size // 2:
            p1, p2 = random.sample(top, 2)
            split = random.randint(1, min(len(p1[0]), len(p2[0])) - 1)
            child = p1[0][:split] + [s for s in p2[0][split:] if s not in p1[0][:split]]
            if child[-1] not in terminal_states:
                child = generate_path()
            offspring.append(child)

        population = [x[0] for x in top] + offspring

    return [(p, fitness(p, transition_times, reward_map)) for p in population]