# undergraduate_thesis

Дипломный проект посвящен анализу процесса найма с использованием методов Process Mining и Reinforcement Learning алгоритмов. 

В проекте используются реальные данные, преобразованные в нужный формат. Анализ заключается в построении графа процесса, выявлении узких мест и моделировании различных сценарии для его оптимизации.

## Установка библиотек и запуск проекта

Чтобы запустить проект, необходимо:

1. Установить Python 3.10

`brew install python@3.10`

`python3.10 --version`

3. Создать виртуальное окружение

`python3.10 -m venv venv`

`source venv/bin/activate  # Linux/macOS`

`.\\venv\\Scripts\\activate  # Windows`

`pip install --upgrade pip setuptools wheel`

4. Установить необходимые библиотеки

`pip install sberpm streamlit pandas numpy matplotlib graphviz gymnasium`

5. Загрузить и распаковать архив по ссылке [https://github.com/argareeva/undergraduate_thesis].

6. Запустить приложение перейдя в нужную папку на своем устройстве:

`cd /путь к папке/app`

`streamlit run app.py`

7. Перейти в браузер по локальному адресу

8. Загрузить тестовые данные. Доступны в текущем репозитории с названием HR_log_obezlich.csv.
