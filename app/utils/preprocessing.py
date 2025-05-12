import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sberpm import DataHolder
import random
from collections import defaultdict

def validate_columns(df):
    """
    Валидирует структуру загруженного датасета. 
    """
    expected_columns = ["case_id", "line_id", "from", "to", "time"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Ошибка: В загруженном файле отсутствуют необходимые столбцы: {', '.join(missing_columns)}.")
        st.stop()
    st.success("Файл успешно загружен и проверен.")


def detect_start_state(df):
    """
    Определяет начальное состояние на основе структуры данных.
    """
    from_states = set(df["from"].unique())
    to_states = set(df["to"].unique())
    start_states = list(from_states - to_states)

    if start_states:
        return start_states[0]
    else:
        return df["from"].unique()[0]


def preprocess_data_with_new_stage(filepath: str) -> DataHolder:
    """
    Загружает CSV, очищает и создаёт DataHolder с временными метками.
    """
    df = pd.read_csv(filepath, sep=";")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['case_id', 'from', 'to', 'time'], inplace=True)
    df = df[df['time'] > 0]

    start_datetime = datetime(2024, 1, 1, 0, 0, 0)
    df['start_time'] = None
    df['end_time'] = None

    for case in df['case_id'].unique():
        case_mask = df['case_id'] == case
        case_df = df[case_mask].sort_values(by='line_id')

        start_time = start_datetime
        start_times = []
        end_times = []

        for time in case_df['time']:
            start_times.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))
            end_time = start_time + timedelta(hours=time)
            end_times.append(end_time.strftime("%Y-%m-%d %H:%M:%S"))
            start_time = end_time

        df.loc[case_mask, 'start_time'] = start_times
        df.loc[case_mask, 'end_time'] = end_times

    case_offsets = {
        case: timedelta(hours=np.random.uniform(0.01, 0.1))
        for case in df['case_id'].unique()
    }
    df['start_time'] = pd.to_datetime(df['start_time']) + df['case_id'].map(case_offsets)
    df['end_time'] = pd.to_datetime(df['end_time']) + df['case_id'].map(case_offsets)

    df.to_csv("data/pm_preprocessed.csv", index=False)

    data_holder = DataHolder(
        data=df,
        col_case='case_id',
        col_stage='to',
        col_start_time='start_time',
        col_end_time='end_time',
        col_duration='time',
        time_format="%Y-%m-%d %H:%M:%S"
    )

    return data_holder