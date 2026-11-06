# 01 — диагностика среды SNDL

Дата: 1 ноября 2026 года, America/Chicago. Первый ограниченный заход.

**Статус: диагностика выполнена; этап 01 не завершён.** Checkout подготовлен, требования и установленная среда проверены. Рабочего запуска learner, проверки повторяемости и восстановления из lake пока нет.

## Вопрос и вывод

Что можно воспроизводимо запускать на компьютере Артура?

Исходники доступны для изучения. Штатный импорт SNDL сейчас останавливается на проверке Python: установлен обычный CPython 3.14.6, тогда как проект требует free-threaded сборку с действительно выключенным GIL. Также не найдены Rust и основные пакеты в этом Python. Это препятствия подготовки среды, а не результат проверки способности learner обучаться.

Для первого небольшого запуска выбран существующий `snn_data_lake.instruments.taught_problems`. Его `main()` проверяет контракт задач и защитные проверки на открытых dev-примерах. По прочитанному коду он не создаёт Brain и не обращается к lake. Эта команда ещё **не выполнена**: сначала нужен совместимый runtime. Она проверит подготовку задач, но сама по себе не докажет успешное обучение S0C0.

## Исходники и версия

- До клонирования `repo/` отсутствовал; в корне были внешний `.git` и `planning/`. Чужие файлы не перезаписывались, внешний Git сохранён.
- Источник: <https://github.com/PVLuanDinh/Spiking-Neural-Data-Lake>.
- Checkout: `/Users/arthurkhvan/Documents/ChatGPT/Spiking-Neural-Data-Lake 2/repo`.
- `HEAD`: `c4c10b285f1741aa798d5ee9c5296ae371fabecf`, дата коммита `2026-10-29`.
- `origin/HEAD` после клонирования указывал на тот же SHA, что указан в рабочем плане.
- Состояние: **detached HEAD**, без рабочей ветки. Для диагностики версия закреплена явно; перед реализацией понадобится отдельная рабочая ветка.
- `git status --short --branch --untracked-files=all` внутри checkout: `## HEAD (no branch)`, без изменений.

Выполненные команды подготовки:

```sh
git clone --no-checkout https://github.com/PVLuanDinh/Spiking-Neural-Data-Lake.git repo
cd repo
git rev-parse --verify 'c4c10b285f1741aa798d5ee9c5296ae371fabecf^{commit}'
git log -1 --format='%H%n%cs%n%s' origin/HEAD
git checkout --detach c4c10b285f1741aa798d5ee9c5296ae371fabecf
git status --short --branch --untracked-files=all
git remote -v
git log -1 --format='%H%n%cs%n%s' -- tools/policy_certification.json
git diff --exit-code HEAD -- src/snn_data_lake/brainstem tools/envelope_hashes.json tools/policy_certification.json
```

Первое скачивание в ограниченной сетевой среде вернуло `Could not resolve host: github.com`; повтор через разрешённый сетевой доступ завершился успешно. Изменения в GitHub не публиковались.

Последний коммит, изменивший `tools/policy_certification.json`, — тот же `c4c10b28…`. В его сообщении записана историческая battery на `ec4df693`. **На компьютере Артура эта battery не запускалась и свежей проверкой не считается.** Чтение `tasks.md` подтверждает открытые строки T110/G2, T111/G3, T113/G5, T114/G6 и T117/G9; их статус здесь не менялся.

## Карта готовности

| Что | Фактический результат | Значение для первого запуска |
|---|---|---|
| ОС | macOS 15.6, build 24G84; Darwin 24.6.0 | Среда определена |
| Архитектура | arm64; MacBook Air Mac14,2, Apple M2 | Нужны совместимые сборки для macOS ARM |
| Память | `system_profiler`: 8 GB | Производительность пока не измерялась |
| Диск | `shutil.disk_usage('.')`: свободно 18 463 891 456 байт на момент проверки | Это снимок доступного места, не оценка стоимости установки |
| `python` | Miniconda, 3.13.13, arm64, GIL включён | Не соответствует runtime проекта |
| `python3` | Python.org, 3.12.2, arm64 | Не соответствует runtime проекта |
| `python3.14` | Homebrew, 3.14.6, arm64, `Py_GIL_DISABLED=0`, GIL включён | Версия языка подходит, сборка — нет |
| Python 3.14t | Не найден в PATH и проверенных стандартных местах | Первый обязательный ресурс |
| Rust / Cargo | `rustc` и `cargo`: command not found; стандартные пути также отсутствуют | Для штатной установки нужно подготовить Rust |
| Закреплённый Rust | `rust-toolchain.toml`: 1.96.0, rustfmt, clippy | Требование исходников, не установленная версия |
| C/C++ compiler | Apple clang 17.0.0, target arm64-apple-darwin24.6.0 | Компилятор найден; сборка проекта не проверялась |
| uv / cmake | Не найдены в PATH; uv также не найден в проверенных стандартных местах | uv — возможный способ получить 3.14t; обязательность cmake для этого Mac не установлена |
| Нативное расширение | В `src/snn_data_lake/` нет `_native.*` | Нужна сборка при штатной установке |
| Изолированный runtime | Не создавался | Подготовка остаётся следующим шагом |

Проверка отсутствия инструментов ограничена PATH и стандартными местами (`~/.cargo/bin`, `/opt/homebrew/bin`, `~/.local/bin`, `/Library/Frameworks/PythonT.framework`). Это не поиск по всему диску.

### Установленные пакеты

Версии получены через `importlib.metadata.version()`, **без импорта самих зависимостей**. Наличие metadata не доказывает работоспособность бинарного пакета. Пакеты Python 3.12 не переносятся автоматически в окружение 3.14t.

| Пакет | `python3` 3.12.2 | `python3.14` 3.14.6 |
|---|---|---|
| torch | 2.12.0 | Не найден |
| numpy | 2.1.1 | Не найден |
| pyarrow | Не найден | Не найден |
| mujoco (проект требует 3.11.0) | Не найден | Не найден |
| brian2 (проект требует 2.10.1) | Не найден | Не найден |
| psutil | 7.2.2 | Не найден |
| threadpoolctl | 3.5.0 | Не найден |
| scipy | 1.15.1 | Не найден |
| networkx | 3.3 | Не найден |
| hatchling / maturin | Не найдены | Не найдены |
| pytest | 9.1.1 | Не найден |
| pytest-cov / pytest-xdist | Не найдены | Не найдены |
| ruff (dev pin проекта: 0.15.20) | 0.16.3 | Не найден |
| nir / h5py | nir отсутствует; h5py 3.15.1 | Не найдены |
| pip | 26.0.1 | 26.1.2 |

В `python` Miniconda 3.13.13 найден pip 26.0.1; остальные перечисленные пакеты не найдены. В каждом интерпретаторе поиск стандартного модуля `json` дал положительный результат. Доступность wheels именно для **macOS arm64 + cp314t**, разрешение зависимостей и сборка Rust на этой машине ещё не проверялись.

## Что фактически проверено

1. Прочитаны рабочий план и статус, единственный найденный `repo/AGENTS.md`, `docs/AGENT_COLD_START.md`, `docs/POLICIES.md`, `SECURITY.md`, `docs/INSTALL.md`, `docs/RUNNING_LOCAL.md`, рабочие разделы upstream-плана и исходники выбранной проверки. Локального `CLAUDE.md` в свежем checkout нет.
2. Через stdlib прочитаны `pyproject.toml`, build-system и зависимости; отдельно прочитаны Cargo manifests, закреплённый toolchain и `infra/native_build.py`.
3. Выполнены `uname -a`, `sw_vers`, `cc --version`, поиск исполняемых файлов и проверки Python. Прямой `sysctl` был запрещён sandbox; сведения об M2 и памяти получены через `system_profiler` с выводом только выбранных полей, без серийного номера.
4. Выполнен один импорт пакета под имеющимся Python 3.14.6. Переменные обхода GIL удалены, bytecode-запись выключена, `NORD_HOME` направлен на отдельный несуществующий путь `planning/scratch/01-diagnostic-import-only`.
5. После диагностики проверены чистота checkout и отсутствие изменений safety envelope и исторической certification-записи.

Команда для повторного чтения характеристик любого из проверенных Python:

```sh
python3.14 -B - <<'PY'
import importlib.metadata as md
import importlib.util
import platform, sys, sysconfig
print(sys.executable, sys.version, platform.machine())
print('Py_GIL_DISABLED:', sysconfig.get_config_var('Py_GIL_DISABLED'))
print('GIL enabled:', getattr(sys, '_is_gil_enabled', lambda: 'not exposed')())
print('json positive control:', importlib.util.find_spec('json') is not None)
for name in ('pip', 'torch', 'numpy', 'pyarrow', 'mujoco', 'brian2', 'psutil',
             'threadpoolctl', 'scipy', 'networkx', 'hatchling', 'maturin',
             'pytest', 'pytest-cov', 'pytest-xdist', 'ruff', 'nir', 'h5py'):
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, 'NOT FOUND')
PY
```

Такая проверка выполнена для `python`, `python3`, `python3.14` через отдельные процессы с таймаутом 20 секунд на интерпретатор. Проверка импорта SNDL выполнялась с таймаутом 15 секунд. Её результат:

```text
returncode: 1
RuntimeError: [import] snn_data_lake requires the FREE-THREADED CPython build
(PEP 703, cp314t) with the GIL genuinely OFF.
Py_GIL_DISABLED=0
sys._is_gil_enabled()=True
expected free-threaded refusal: True
diagnostic NORD_HOME exists after import: False
native extensions: []
```

Это ожидаемое срабатывание защиты в `snn_data_lake.__init__.require_free_threaded()`, а не успешно пройденный self-check. Обход `NORD_ALLOW_GIL=1` не использовался. Исходники пакета при импорте не изменились.

## Небольшая проверка и полный Brain

У проекта нет поддерживаемого режима Brain без Torch. В `pyproject.toml` обязательны все девять core-зависимостей из таблицы выше. `pip install .` использует Hatchling и maturin; `infra/native_build.main()` собирает расширение и проверяет free-threaded ABI. Простая замена номера версии Python проблему не решает.

Для учебной проверки найден более узкий вход: `instruments/taught_problems.py::main`. Он вызывает B9 (критерий решения), B11 (бюджет), B12 (чтение без изменения learner) и специально испорченные варианты этих проверок. В исходниках этой проверки используются уже открытые dev seeds 8000 и 8900; новые confirmatory/reserve seeds не открывались.

Карта существующего пути по исходникам, **без подтверждения запуском**:

| Шаг | Файл и функция |
|---|---|
| Входные эпизоды и целевые ответы | `instruments/delayed_cue.py::episode_i`, `label_i`; `instruments/taught_problems.py::Problem.teach_epochs`, `Problem.episodes` |
| Текст → числовые признаки | `readout/interaction_learning.py::text_features` (CRC32 слов) |
| Признаки → времена импульсов | `SpikingInteractionLearner.encode` → `sensorimotor/rank_order_coding.py::rank_code` |
| Обучение существующего learner | `taught_problems.canonical_incumbent` создаёт `SpikingInteractionLearner` с `PlasticityRuleStack`; `learn_turn` / `interact` / `_deposit` обновляют R-STDP нейроны |
| Readout ответа | `SpikingInteractionLearner.route_text` / `respond` / `_argmax_label` |
| Проверка с выключенным обучением | `taught_problems.read_split`, `solved`, `check_b12`; общий ограниченный вход — `taught_problems.main` |

Этот learner сам содержит читающие ответ нейроны. Он не равен полному маршруту «Brain → reservoir → readout → возврат в Brain». `taught_campaign.Reader` и `attempt` найдены как существующие средства для дальнейшего разбора, но кампания не запускалась. Голый запуск `readout.interaction_learning` не выбран: его self-check включает интеграционный путь с Brain.

В `models/` Git отслеживает только `nmnist_pack.npz` и `shd_pack.npz`. Работоспособность внешних perceptual checkpoints, CUDA/MPS, полного Brain и факультетов с дополнительными моделями не проверялась. Для выбранного входа `taught_problems.main` внешние perceptual checkpoints не используются по прочитанному коду; это ещё не проверка исполнения.

Найденное расхождение документации: `docs/RUNNING_LOCAL.md` обещает zero-dependency core и запуск `run_selfchecks.py` сразу после клонирования. Это расходится с `AGENTS.md`, `pyproject.toml` и фактическим отказом импорта. Старую инструкцию установки здесь не применяли и upstream-документы не редактировали. В `AGENT_COLD_START.md` §8.8 также сохранился старый фиксированный S=8; для будущих научных кампаний приоритет имеет актуальная поправка P1 о мощности и последовательном дизайне.

## Следующий ограниченный шаг — пока не выполнен

Сначала подготовить отдельный CPython 3.14t **вне `repo/`** и убедиться, что проверка показывает `Py_GIL_DISABLED=1` и `sys._is_gil_enabled()=False`. Затем подготовить Rust 1.96.0 и проверить доступность обязательных пакетов для этого Mac, прежде чем устанавливать core. Не копировать пакеты из Python 3.12.

Первая конкретная команда подготовки, из корня рабочего проекта:

```sh
python3.14 -m venv .venv-01-bootstrap
```

В последующем заходе этот вспомогательный venv можно использовать для отдельной установки uv; целевой запрос интерпретатора из текущего сообщения защиты — `uv python install 3.14+freethreaded`. uv пока отсутствует. Эти команды подготовки в данном заходе не выполнялись.

После создания совместимого окружения `.venv-01` рядом с `repo/` и штатной установки core следующая команда первого учебного self-check будет такой:

```sh
cd "/Users/arthurkhvan/Documents/ChatGPT/Spiking-Neural-Data-Lake 2/repo"
env -u NORD_ALLOW_GIL -u PYTHON_GIL \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 PYTHONPATH=src \
  ../.venv-01/bin/python -B - <<'PY'
import os, subprocess, sys, sysconfig, tempfile
assert sysconfig.get_config_var('Py_GIL_DISABLED') == 1
assert sys._is_gil_enabled() is False
with tempfile.TemporaryDirectory(prefix='sndl-01-') as task_home:
    child_env = os.environ.copy()
    child_env['NORD_HOME'] = task_home
    subprocess.run(
        [sys.executable, '-B', '-m', 'snn_data_lake.instruments.taught_problems'],
        env=child_env, check=True, timeout=60,
    )
PY
```

Команда приведена по прочитанному коду, **не как уже рабочая на этом компьютере**. Ожидаемый признак успеха — exit code 0 и строка с префиксом `self-check OK: B9, B11, B12;`. Таймаут — предел попытки, не измеренное время исполнения. Для проверки повторяемости нужно будет повторить её в новом процессе и сохранить оба полных выхода отдельно. После этого можно готовить учебный показ обучения до/после на S0C0; его результат здесь не предсказан.

## Границы и оставшаяся работа

Не выполнялись установки зависимостей, сборка native, learner self-check, обучение, battery, полная pytest-suite, daemon, создание Brain, научные кампании или запись в lake. Live home, физическая рука и safety envelope не использовались. Пробный `NORD_HOME` после отказа импорта не появился. Не создавались другие чаты, не выполнялась делегация и не отправлялись сообщения в другие чаты.

Для завершения этапа 01 остаются совместимый runtime и core, успешный небольшой запуск, повторение в новом процессе и сохранённые результаты. Совместимость пакетов на macOS ARM ещё предстоит установить. Общий `planning/STATUS.md` этим чатом не изменялся. Checkout оставлен чистым на закреплённом SHA; отчёт находится снаружи checkout. Коммиты и публикации не создавались.
