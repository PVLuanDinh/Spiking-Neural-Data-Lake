# 01 — среда и первый ограниченный запуск

Дата: 1 ноября 2026 года, America/Chicago. Продолжение после первого диагностического захода.

**Статус этапа 01: готов для минимального маршрута.** Подготовлен изолированный Python 3.14t; штатная ограниченная проверка задач прошла и повторилась в новом процессе с побайтно одинаковым выводом. Полный Brain не устанавливался и не запускался. Успешное обучение на S0C0 этим результатом ещё не показано.

## Вопрос и результат

Что можно воспроизводимо запускать на компьютере Артура?

На Mac M2 arm64 работает существующий self-check `snn_data_lake.instruments.taught_problems` прямо из закреплённых исходников. Это небольшая проверка подготовки задач и защит от неправильной оценки. Для неё фактически потребовался только free-threaded Python: в целевом venv установлен один pip, внешние пакеты не установлены. Это измерение узкого входа, не поддерживаемый режим Brain без Torch и не полноценная установка пакета.

Обнаруженное в первом заходе препятствие с Python устранено. Результаты первой диагностики сохранены без изменений в [архиве](01-artifacts/01-diagnostics-first-pass.md).

## Версия и конфигурация

- Исходники: `repo/`, origin `https://github.com/PVLuanDinh/Spiking-Neural-Data-Lake.git`.
- `HEAD`: `c4c10b285f1741aa798d5ee9c5296ae371fabecf`, detached HEAD; рабочие исходники не изменялись.
- ОС и оборудование из диагностики: macOS 15.6, arm64, Apple M2, 8 GB памяти.
- Вспомогательное окружение: `.venv-01-bootstrap`, обычный Python 3.14.6, uv 0.13.6.
- Учебное окружение: `.venv-01`, **CPython 3.14.6 free-threading build**, arm64, `SOABI=cpython-314t-darwin`.
- Проверено: `Py_GIL_DISABLED=1`, `sys._is_gil_enabled()=False` до запуска; GIL остался выключен после self-check.
- Managed Python находится в `.runtime-01/python/`; кэш загрузок — в `.cache-01/`. Эти каталоги и оба venv находятся снаружи checkout и исключены из внешнего Git через корневой `.gitignore`.
- Системный Python и shell PATH не менялись. Использован `uv python install --no-bin`; системные ссылки на Python не создавались.

## Команда повторения

Выполнить из корня рабочего проекта:

```sh
cd "/Users/arthurkhvan/Documents/ChatGPT/Spiking-Neural-Data-Lake 2"
.venv-01/bin/python -B planning/reports/01-artifacts/run-taught-check.py
```

[Скрипт повторения](01-artifacts/run-taught-check.py) создаёт отдельный временный `NORD_HOME`, удаляет переменные обхода GIL, задаёт `PYTHONPATH=repo/src` и запускает модуль в дочернем процессе с лимитом 60 секунд. Bytecode-запись выключена. После выполнения проверяет, что GIL всё ещё выключен и временный home пуст. При ошибке или таймауте команда завершается неуспешно.

## Фактически выполненные проверки

Сначала модуль успешно запущен напрямую в отдельном процессе; выход сохранён как `source-smoke-initial.*.txt`. Затем скрипт повторения выполнен два раза, каждый раз с новым дочерним процессом и новым временным home.

| Проверка | smoke-1 | smoke-2 |
|---|---|---|
| Exit code | 0 | 0 |
| Проверки B9, B11, B12 | OK | OK |
| Специально испорченные случаи | 5 из 5 обнаружены | 5 из 5 обнаружены |
| GIL после self-check | False | False |
| Изолированный home | Пуст | Пуст |
| stderr | Пуст | Пуст |

Полный stdout совпал побайтно. SHA-256 обоих выходов:

```text
d6d5e69040526bc2ecf18e8271ac817de2fe7cad1493dc905f3270244d87d22e
```

Записи результатов:

- [Параметры среды, версия кода, хеш скрипта и результаты повторов](01-artifacts/smoke-results.json).
- [Первый повтор: stdout](01-artifacts/smoke-1.stdout.txt), [stderr](01-artifacts/smoke-1.stderr.txt).
- [Второй повтор: stdout](01-artifacts/smoke-2.stdout.txt), [stderr](01-artifacts/smoke-2.stderr.txt).

Сравнивался вывод **этой проверки**. Побайтная повторяемость всего Brain, качество обучения и сохранение наученного состояния в lake не проверялись.

### Что означают B9, B11 и B12

- B9 проверяет критерий решения: пустой или неполный/неверный ответ не должен считаться решением.
- B11 проверяет правила бюджета и его нижнюю границу; успешная проверка не обещает, что бюджета достаточно для обучения.
- B12 проверяет, что проверочное чтение не изменяет состояние learner. Отдельный намеренно неправильный вариант вызывает обучение, и защита должна это обнаружить.

В self-check используются уже открытые dev seeds 8000 и 8900. Новые научные cohorts и confirmatory/reserve seeds не открывались. Контрольный вызов обучения внутри испорченного B12 — часть проверки защиты, не эксперимент по эффективности обучения.

## Карта пути по существующему коду

Пути в таблице относятся к `repo/src/snn_data_lake/`.

| Шаг | Файл и символ |
|---|---|
| Входные эпизоды и ответы | `instruments/delayed_cue.py::episode_i`, `label_i`; `instruments/taught_problems.py::Problem.teach_epochs`, `Problem.episodes` |
| Текст → числовые признаки | `readout/interaction_learning.py::text_features`, CRC32 слов |
| Признаки → времена импульсов | `SpikingInteractionLearner.encode` → `sensorimotor/rank_order_coding.py::rank_code` |
| Существующий learner | `taught_problems.canonical_incumbent` создаёт `SpikingInteractionLearner` с `PlasticityRuleStack` |
| Изменение весов | `SpikingInteractionLearner.learn_turn` / `interact` / `_deposit`, R-STDP нейроны |
| Readout ответа | `SpikingInteractionLearner.route_text` / `respond` / `_argmax_label` |
| Оценка без обучения | `taught_problems.read_split`, `solved`, `check_b12` |
| Проверенный вход | `taught_problems.main` |

Сквозная карта обучения приведена по коду; запущен именно `main()` с его защитными проверками. Этот learner содержит читающие ответ нейроны; он не равен полному пути «Brain → reservoir → readout → возврат в Brain». Существующие `taught_campaign.Reader` и `attempt` найдены для следующего разбора, но кампания не запускалась. Прямой запуск `readout.interaction_learning` не используется: его self-check включает Brain.

## Готовность полного пакета и оставшиеся зависимости

| Ресурс | Текущее состояние |
|---|---|
| Python 3.14t | Установлен и проверен для узкого входа |
| Rust 1.96.0 / Cargo | Не установлены; нужны для штатной сборки пакета |
| Нативное расширение `_native` | Не собрано; для выполненного self-check не понадобилось |
| Девять обязательных core-зависимостей | Проверено разрешение совместимых готовых пакетов; не установлены |
| Полный Brain и внешние модели | Не проверены; perceptual checkpoints не готовились |
| CUDA/MPS и ускорение | Не проверены |

Resolver uv успешно подобрал **25 пакетов** с транзитивными зависимостями для этого интерпретатора при запрете сборки исходных дистрибутивов (`--only-binary :all:`). Это успешный подбор metadata, ещё не проверка установки или импорта бинарных библиотек. Зафиксированы, в частности: torch 2.15.0, numpy 2.5.4, pyarrow 26.0.0, mujoco 3.11.0, brian2 2.10.1, psutil 7.2.2, threadpoolctl 3.7.0, scipy 1.18.1, networkx 3.7.

[Зафиксированные зависимости с хешами](01-artifacts/core-requirements.txt) и [выход resolver](01-artifacts/core-resolution.log) сохранены для дальнейшей подготовки полного пакета. Необходимость устанавливать их для текущей проверки отпала после успешного запуска исходников. Штатная установка полного пакета по-прежнему требует core и Rust; обходов этого требования не добавлялось.

## Выполненные команды подготовки

Все команды ниже выполнены из корня проекта. Для pip/uv-загрузок использован разрешённый сетевой доступ.

```sh
python3.14 -m venv .venv-01-bootstrap
.venv-01-bootstrap/bin/python -m pip install uv --cache-dir .cache-01/pip --disable-pip-version-check
UV_PYTHON_INSTALL_DIR="$PWD/.runtime-01/python" UV_CACHE_DIR="$PWD/.cache-01/uv" .venv-01-bootstrap/bin/uv python install 3.14.6+freethreaded --no-bin --no-config --no-progress
UV_PYTHON_INSTALL_DIR="$PWD/.runtime-01/python" UV_CACHE_DIR="$PWD/.cache-01/uv" .venv-01-bootstrap/bin/uv venv --seed --python 3.14.6+freethreaded .venv-01 --no-config --no-progress
UV_CACHE_DIR="$PWD/.cache-01/uv" .venv-01-bootstrap/bin/uv --no-config pip compile repo/pyproject.toml --python .venv-01/bin/python --only-binary :all: --generate-hashes --no-progress --output-file planning/reports/01-artifacts/core-requirements.txt
.venv-01/bin/python -B planning/reports/01-artifacts/run-taught-check.py
```

Скрипт запускался дважды через `subprocess.run` с захватом stdout/stderr. Контрольная сводка проверила exit code, ожидаемую строку успеха, пустой stderr и равенство stdout. Таймаут 60 секунд — ограничение попытки, не измерение производительности.

Прочитанные применимые инструкции, результаты диагностики Python/Rust, историческая battery и расхождения старых документов подробно перечислены в архиве первого захода. Исторические battery не считаются выполненными здесь. В частности, обещание zero-dependency core в старом `RUNNING_LOCAL.md` не является описанием полного текущего пакета.

## Границы результата и передача дальше

Полный Brain, daemon, физическая рука, safety envelope и live home не использовались. В lake ничего не записано. Исходники, исторические научные результаты и certification-запись не изменены; GitHub не обновлялся. Другие чаты не создавались, делегации и сообщений в другие чаты не было. Общий `planning/STATUS.md` остаётся за координационным чатом и здесь не редактировался.

Критерий этапа 01 выполнен в разрешённой минимальной форме: есть рабочая команда штатной ограниченной проверки, независимое повторение и список ресурсов для полного пакета. Следующий учебный шаг относится к этапу 02: показать вход, целевой ответ и измерение до/после обучения на существующей задаче S0C0, используя этот runtime. Эффект обучения, readout полного двигателя, сон и восстановление навыка остаются непроверенными.
