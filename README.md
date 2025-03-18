# undergraduate_thesis

Дипломный проект посвящен анализу процесса найма с использованием библиотеки SberPM. 

В проекте используются реальные данные, преобразованные в нужный формат, что позволяет строить графы процесса, выявлять узкие места и моделировать различные сценарии для его оптимизации.

## Установка SberPM на macOS

Чтобы запустить проект, необходимо установить SberPM и Graphviz (пример установки на macOs).

1. Установить Python 3.10

`brew install python@3.10`

`python3.10 --version`

3. Создать виртуальное окружение

`pip install --upgrade pip setuptools wheel`

`pip install sberpm graphviz jupyter`

4. Установить Graphviz

`brew install graphviz`

`dot -V`

6. Подключить виртуальное окружение к Jupyter

`pip install ipykernel`

`python -m ipykernel install --user --name=sberpm_env --display-name "Python (SberPM)"`

`jupyter notebook`
