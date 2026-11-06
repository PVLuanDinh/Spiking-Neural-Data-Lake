# 06 — длительность offline-обработки и один pruner

2 ноября 2026 года, America/Chicago.

**Статус: локальные опыты с длительностью и P-LL выполнены. Сравнение стадий
NREM/REM на baseline 05 не выполнено: отсутствует совместимый интерфейс.**

На четырёх открытых development seeds один смешанный проход replay уже восстановил
старый класс A, сохранив новый B. Отдельная штатная операция pruning сократила
12 прототипов до 6, сохранила все три синтетические задачи и вдвое сократила поиск.
Структура, память и ответы восстановились в новых процессах.
Эти два результата относятся к разным learner’ам; pruning readout этапа 05
или всего Brain ими не проверен. Полный этап со сравнением стадий остаётся открытым.

## Вопрос, зависимость и исходники

Вопрос: что меняется при увеличении offline replay и что реально экономит одна
выбранная операция удаления, если отдельно проверять старые и новые задачи?

Зависимость 05 проверена: использован выбранный там checkpoint **b30** после
забывания A из-за последовательного обучения классов A_c1d: A=GREET, B=GOAL.
Данные, encoding, delta-rule и порядок train-предъявлений сохранены. Конечные
полные состояния при 0 и 10 проходах совпали с none/replay этапа 05 у каждого seed.

Checkout `repo/` до/после чистый, detached HEAD
**c4c10b285f1741aa798d5ee9c5296ae371fabecf**. Код, исторические протоколы/seeds,
сертификация, ветки и коммиты не менялись. Обвязка и записи находятся вне `repo/`.
Прочитаны AGENTS, cold-start, SECURITY, POLICIES P0–P8/invariants и актуальные
ограничения G6; это отдельный локальный diagnostic, не исполнение upstream G6.

[Протокол до task data](06-artifacts/protocol.md),
[журнал](06-artifacts/LEDGER.md),
[manifest с командами и provenance](06-artifacts/pilot-1/manifest.json).

## Сначала census

Фактически запущен штатный `sleep_pruner_census.census_static()` и его brain-free
self-check. Получены пять типов:

| ID | Операция | Статически найденный путь |
|---|---|---|
| P-DP | `AttractorNet.downscale_prune` | `Brain._sleep_tick` |
| P-SR | `StructuralRewiring.rewire` | `_sleep_tick` → R2 sleep step |
| P-EY | `NGSNN.prune_unearned` | `_sleep_tick` → grower refit |
| P-LL | `LifelongConsolidator._prune_to_budget` | `_enter_sleep` → `learn_task` |
| P-LAKE | `LakeConsolidation/KnowledgeLog.consolidate` | `_enter_sleep` |

Десять vocabulary sites включают bookkeeping и inserts, которые census отдельно
классифицирует как не-pruner. Учтены **394 неразрешённых receiver calls**;
неразобранных hits именно prune-vocabulary и пропавших ожидаемых типов нет.
Проверки с добавленным/удалённым вызовом и planted work-counter прошли.

Это **статический census, не runtime census полного Brain**. Brain не создавался,
его естественные sleep cycles не измерялись. Приоритеты scopes соблюдены:
камера/восприятие отложены; CLI исторической G6 campaign с её seed block не запускался.
Фактические вызовы выбранного P-LL измерены ниже штатным `Recorder/counting_patches`.

[Полный census и stdout self-checks](06-artifacts/census.stdout.json).

## Опыт 1: длительность, без смены механизма

Каждому arm дана копия того же b30 lake. Seeds **8000–8003** уже раскрыты в 02–05;
это не новый confirmatory cohort. Рост и reservoir заморожены. Исходная конфигурация
05 сохранена: KernelLsmPool 256, carry внутри эпизода/reset на границе, TUNED,
frac_bits=12, P=1, threads=1; DeltaAdaptor/OnlineReadout mu=1.

Replay использует только 32 train-примера, в тех же смешанных порядках.
Все объявленные бюджеты выполнены; остановки или отбора по final нет.
На класс — 16 разных train, 4 dev и 6 final текстов. Повторённые final-предъявления
дают 24 ответа на класс, а не 24 независимых текста.

**На каждом seed одинаково:**

| Проходы replay | Delta updates | A final | B final |
|---:|---:|---:|---:|
| 0 | 0 | 0/24 | 24/24 |
| 1 | 32 | 24/24 | 24/24 |
| 2 | 64 | 24/24 | 24/24 |
| 5 | 160 | 24/24 | 24/24 |
| 10 | 320 | 24/24 | 24/24 |

При 10 проходах label-shuffle с сохранением количества меток дал A
**24, 4, 16, 0 из 24**, B **20, 8, 0, 16 из 24** в порядке seeds 8000–8003.
Чтение чувствительно к обучающему сигналу; отдельный seed с высокой случайной
точностью не выбран как результат метода.

CPU только replay-loop: один проход **1.604–1.999 ms**, десять
**8.404–9.046 ms**. Импорты, feature extraction, save/restore и проверка сюда не
входят. Это отдельные короткие измерения с startup/warmup влиянием, не линейная
модель времени: например, два прохода иногда заняли меньше первого.

**Вывод:** один полный проход — минимальный из проверенных положительных бюджетов.
Дополнительные проходы не повысили уже предельную точность. Бюджеты между 0 и 32
updates, loss/калибровка, устойчивость к новым помехам и другие задачи не оценены;
«одного прохода всегда достаточно» из этого не следует.

## Стадии: конкретное ограничение

Штатный `staged_sleep_bout` работает с `SpikingInteractionLearner`: NREM повторяет
его interaction replay, REM делает synthetic forward и обращается к его rest-stack.
У delta baseline этапа 05 нет необходимых `neurons`, `_replay`, `_deposit`,
`interact`, `encode`, `_argmax_label`, `stack`. Проверка этого контракта выполнена
в controls-worker; статус **NOT_EVALUATED** сохранён в машинной сводке.

Brain-free self-check runner прошёл: accounting, исключение held-out из replay,
fabrication guard, starvation и planted compute mismatch. Это проверка инструмента,
а не результат сна. Его `check_compute` сравнивает realized presentations;
NREM deposits и REM forward/rest не становятся равными CPU-операциями от этого.

Пакетный solve этапа 05 не переименован в REM; retired automatic replay/rest/attractor
sleep не возвращались. Для реального сравнения нужен отдельно заданный совместимый
learner и механизм стадий, затем сопоставление измеренных затрат. Это открытая
зависимость, а не отрицательный результат эффективности стадий.

## Опыт 2: P-LL, отдельный learner и одна операция

Выбран существующий `LifelongConsolidator._prune_to_budget`: он физически удаляет
прототипы из активного списка. Это nearest-prototype classifier, не reservoir
и не readout 05. Штатный `_make_tasks` задаёт **3 задачи × 2 класса**, dim=36.
У каждого класса есть частый и редкий близкие modes; избыточность встроена в fixture.
На задачу — 32 train и 40 test векторов; train/test полные векторы не пересекаются.
Здесь A/B/C — три другие задачи, не GREET/GOAL из первого опыта.

Для каждого открытого seed обучены все три задачи с replay=True, prune=False,
budget=6, epochs=2, остальные defaults. После этого **из одного состояния с 12
прототипами** выполнены четыре независимых arm. После развилки обучение выключено.
Это одно post-training удаление, не сравнение траекторий с pruning после каждой задачи.

| Arm | Активных прототипов | A | B | C |
|---|---:|---:|---:|---:|
| OFF | 12 | 40/40 | 40/40 | 40/40 |
| Удалить наименьшую полезность | 6 | 40/40 | 40/40 | 40/40 |
| Штатный random, столько же удалений | 6 | см. ниже | см. ниже | 40/40 |
| Удалить наибольшую полезность | 6 | 40/40 | 40/40 | 40/40 |

OFF, magnitude и highest-utility дали строки таблицы **на всех четырёх seeds**.
Random потерял один класс: A=20/40 у 8000/8002, B=20/40 у 8001/8003; остальные
задачи дали 40/40. Один random draw на seed предусмотрен заранее, удачные маски
не отбирались. До развилки совместное решение всех задач достигнуто.

Штатный счётчик подтвердил по одному вызову на arm: OFF удалил 0, magnitude и
random — по 6. Highest-utility — внешний adversarial control, native call ему
не приписывается. Он тоже сохранил все классы: в этом fixture можно оставить
и частый, и редкий представитель класса. Следовательно, преимущество именно
низкой utility перед высокой **не показано**. Это ограничение атрибуции нельзя
скрыть за выигрышем против random; deciding G6 R2/KEEP не подтверждён.

## Реальные затраты и lake

Счётчик на фактическом `_cosine` измерил **12 → 6 сравнений** и **432 → 216
просматриваемых координат** на один запрос. Здесь экономия есть потому, что
поиск обходит более короткий список. Это не подсчёт всех CPU-инструкций и не
утверждение об энергии.

Пять блоков по 360 запросов на arm, threads=1; ниже медианы CPU на запрос:

| Seed | OFF, μs | Magnitude, μs | Отношение времени |
|---|---:|---:|---:|
| 8000 | 27.594 | 13.858 | 1.991 |
| 8001 | 28.242 | 13.883 | 2.034 |
| 8002 | 27.639 | 13.683 | 2.020 |
| 8003 | 27.981 | 14.728 | 1.900 |

Отдельный выбор маски magnitude занял **29–44 μs CPU**, включая wrapper счётчика.
Рассмотрены 12 кандидатов-прототипов, один фиксированный бюджет; поиска
гиперпараметров и отбора по test нет. Clone/checkpoint/rebuild/import затраты
в query timing не входят; сравнение всего Brain или end-to-end ускорение не заявляются.

Pruning **не удалял старые записи lake**: каждый arm сохранил исходный byte-prefix
и append принятых прототипов, labels, utilities, config, RNG и train streams.
Штатные `state_dict/load_state_dict`, `rebuild_memory` и `save/load_named_state`
использованы без новых тегов. Memory восстанавливается из train streams внутри
lake с точной позицией sampler; оба RNG сравниваются. Рабочая структура — часть
named-state, а не JSON-sidecar.

Magnitude lake: **173 222–174 183 bytes**, OFF: **177 555–178 509 bytes**.
Это общий файл с исходной и новой записью. Удаление шести активных прототипов
не означает освобождения половины дискового файла; compaction не выполнялась.

## Проверки, стоимость всего блока и воспроизведение

- **42 пары writer/restore** совпали по полным состояниям и всем ответам; writer
  завершался до чтения нового процесса. Сюда включены два повтора 8000; они не
  увеличивают N. Для duration вновь построен pool и сверены параметры/признаки по
  текстам без целей. Для pruning restore получает только test x, без test y.
- При оценке запрещены learn/observe/solve, сравниваются learner/replay hashes и
  байты lake. Offline-reader независимо пересчитал ответы из beta и прототипов,
  scores, mask alignment, update/call/work counters и разобрал named-state frames.
- До task data прошли прежние controls 05 и **9 новых проверок**: ranking,
  matched-random, чувствительный sabotage на отдельном fixture, missing/stale,
  profile/checksum, width и запрещённое обучение при чтении. Ещё **10 искажений**
  записей отвергнуты offline-reader; его positive прошёл.
- Два пересчёта побайтно совпали по summary/sensitivity/table. SHA256 summary:
  `7ebdfe5b3a3982aebb81b440c5aa104f3a5e6b22a6cf709c5e0094965bce111b`.
  [Итоговая сверка](06-artifacts/closing-verification.json).

Блок: **243.116 s wall**, **89 workers**, сумма worker wall **221.865 s**;
CPU рабочих функций **7.390 s**, max RSS одного worker **264 011 776 bytes**,
stdout **4 957 700 bytes**. Function CPU исключает основную часть импортов/startup;
block wall включает subprocess orchestration и копирование артефактов. Подготовка,
census и offline-аудиты вне этой суммы. Финансовые расходы не измерялись.

CPython **3.14.6 free-threading**, Torch **2.15.0**, NumPy **2.5.4**, GIL off
после работы всех workers. Использована прежняя временная runtime-site 05;
проверены реальные пути импортов и версии, новые зависимости не устанавливались.
Системный Python 3.9 использован только для orchestration/offline-reader.
Все **483 src/tools Git blobs** временного source export сверены до/после;
исходные четыре lake 05 не изменились. Runtime dylib имела macOS dataless flags:
первая подготовительная попытка завершилась 137 до source receipts; успешный
census-worker занял 118.369 s. Это не task failure.

Фактически выполнено из корня проекта:

```sh
/usr/bin/python3 -B planning/reports/06-artifacts/experiment.py --run-id pilot-1 --source-root /private/tmp/sndl-class-source-tlefts_z --runtime-site /private/tmp/sndl-class-packages-749rpbye/site
/usr/bin/python3 -B planning/reports/06-artifacts/analyze.py --records planning/reports/06-artifacts/pilot-1 --output planning/reports/06-artifacts/audit-1
/usr/bin/python3 -B planning/reports/06-artifacts/analyze.py --records planning/reports/06-artifacts/pilot-1 --output planning/reports/06-artifacts/audit-2
```

Повтор требует новых run-id/output; старые каталоги не перезаписываются.
Временные source/runtime пути может удалить ОС; их receipts сохранены, для нового
окружения нужны тот же Git export и те же библиотеки. Offline-пересчёт требует
только Python и сохранённых записей 05/06, не Torch или нового обучения.

[Все результаты по seeds](06-artifacts/audit-1/table.md),
[машинная сводка](06-artifacts/audit-1/summary.json),
[post-data изменения только offline-аудита](06-artifacts/audit-amendment.md).
Training harness и условия опыта после данных не менялись.

## Ограничения и следующий шаг

Результат P-LL полезен в узком смысле: на штатном избыточном fixture есть
сокращение поиска без потери проверенных задач. Улучшения точности против OFF нет;
превосходство utility-ranking над highest-utility control не установлено.
Это не решение KEEP/RETIRE и не предложение переносить pruner на reservoir 05.

N=4 — раскрытый exploratory набор; мощности, population effect, MDE и эквивалентности
методов не заявляем. Нулевая SD наблюдаемых точностей не означает известной нулевой
вариативности. P1 требует нового расчёта мощности/sequential plan и lawful untouched
seeds до подтверждающей кампании; P8 не разрешает удаление кода по этому чтению.

Открыты: совместимый опыт со стадиями при измеренных сопоставимых затратах;
pruning reservoir/readout 05; производственные Brain flush/R2 crash continuation;
интеграция роста/feed/return и всего пластичного Brain. Полная battery/pytest,
GPU/native parity и независимый panel не выполнялись. Выполнены три self-review
прохода, [ограничения и решения](06-artifacts/review.md).
Live home для обучения/сохранения, daemon, физическая рука, safety envelope,
GitHub, другие чаты, делегации и мониторинг не использовались. Общий STATUS
оставлен координационному чату; фактический статус этого этапа указан в начале отчёта.

Следующее конкретное действие: отдельно зафиксировать выбор learner и механизма
стадий для сравнения NREM/REM; не переносить сюда исторический результат F5 и не
объявлять полный этап 06 завершённым до этой проверки.
