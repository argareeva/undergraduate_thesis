import graphviz
from pathlib import Path
from sberpm.imitation import Simulation
from sberpm import DataHolder
from sberpm.miners import SimpleMiner
from sberpm.metrics import ActivityMetric
from sberpm.visual import Graph, GraphvizPainter

def generate_and_prepare_holder(simulation, iterations=1000):
    """
    Генерирует данные из симуляции и возвращает DataHolder.
    """
    simulation.generate(iterations=iterations)
    data = simulation.get_result()
    holder = DataHolder(
        data=data,
        col_case="case_id",
        col_stage="to",
        col_start_time="start_time",
        col_end_time="end_time",
        col_duration="time",
        time_format="%Y-%m-%d %H:%M:%S"
    )
    return data, holder


def render_graph(data_holder, filename="graph", format="png"):
    """
    Строит и сохраняет граф процесса на основе метрик и SimpleMiner.
    """
    activity_metric = ActivityMetric(data_holder, time_unit="h")
    nodes_count_metric = activity_metric.count().to_dict()
    nodes_mean_metric = activity_metric.mean_duration().to_dict()

    simple_miner = SimpleMiner(data_holder)
    simple_miner.apply()
    graph = simple_miner.graph

    graph.add_node_metric('count', nodes_count_metric)
    graph.add_node_metric('mean_duration', nodes_mean_metric)

    painter = GraphvizPainter()
    painter.apply(graph)
    
    output_dir = Path("saved_graphs")
    output_dir.mkdir(exist_ok=True)
    graph_path = output_dir / f"{filename}.{format}"

    painter.write_graph(str(graph_path), format=format)
    return str(graph_path)


def show_graph(graph: Graph, save=True, filename="graph", **kwargs):
    painter = GraphvizPainter()
    painter.apply(graph)

    if kwargs.get("happy_path"):
        painter.apply_happy_path(graph, kwargs["happy_path"])
    elif kwargs.get("auto_insights"):
        painter.apply_insights(graph, **kwargs["auto_insights"])
    else:
        painter.apply(graph, **kwargs)

    if save:
        output_dir = Path("saved_graphs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / f"{filename}.png"
        painter.write_graph(str(filepath), format="png")
        return str(filepath)

    return None


def analyze_and_show(holder, success_state=None, title="graph", label=""):
    """
    Анализирует процесс, сохраняет граф и возвращает метрики + путь к картинке.
    """
    miner = SimpleMiner(holder)
    miner.apply()

    activity_metric = ActivityMetric(holder, time_unit="h")
    nodes_count_metric = activity_metric.count().to_dict()
    nodes_mean_metric = activity_metric.mean_duration().to_dict()

    graph = miner.graph
    graph.add_node_metric("count", nodes_count_metric)
    graph.add_node_metric("mean_duration", nodes_mean_metric)

    # Сохраняем картинку
    graph_path = show_graph(graph, filename=title, node_style_metric="mean_duration", edge_style_metric="mean_duration")

    # Метрики
    avg_time = holder.data["time"].mean()
    avg_case_duration = holder.data.groupby("case_id")["time"].sum().mean()

    successful_case_ids = holder.data[holder.data["to"] == success_state]["case_id"].unique()
    successful_cases_duration = holder.data[holder.data["case_id"].isin(successful_case_ids)]
    real_success_avg = successful_cases_duration.groupby("case_id")["time"].sum().mean()

    return {
        "graph_path": graph_path,
        "avg_time": avg_time,
        "avg_case_duration": avg_case_duration,
        "real_success_avg": real_success_avg
    }