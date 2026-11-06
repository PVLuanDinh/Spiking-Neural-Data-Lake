# Этап 06 — журнал

2 ноября 2026 года. Локальный диагностический режим: shell доступен,
upstream checkout остаётся read-only. Задача Артура — начать этап 06 после 05.

Цель: раздельно оценить длительность offline replay, применимость стадий сна и
один штатный pruner, с controls, удержанием старого/нового и рестартом.

Проверено чтением: отчёт 05 и его код; checkout чистый на c4c10b28;
AGENTS, cold-start, SECURITY, POLICIES P0–P8/invariants; актуальные ограничения
G6 A2–A4 и существующие census, staged runner, LifelongConsolidator.
Исходники не изменяются. Ветка не нужна для внешней экспериментальной обвязки.
Полная upstream battery, исторические G6 cohorts и seeds не открываются.

Открытые вопросы:
- Подтверждает ли вычисленный census выбранный путь P-LL?
- Достаточно ли короткого replay для восстановления A из checkpoint b30?
- Есть ли у baseline 05 интерфейс для штатного REM/rest?
- Сокращает ли P-LL поиск без потери старых/новых задач на штатном fixture?
- Восстанавливаются ли структура, память, RNG и ответы в новом процессе?

Проверено запуском: все 483 Git blobs source export совпали. Static census дал
P-DP/P-EY/P-LAKE/P-LL/P-SR; 394 unresolved receivers, ноль unclassified prune hits.
Штатные brain-free self-check census и staged runner прошли с planted reds.
Это не runtime census полного Brain. Первый запуск CPython завершился 137 до
source receipts; повтор с системным Python-оркестратором успешен, worker 118.369 s.
У runtime dylib наблюдались compressed,dataless flags; не считать это task failure.

Выполнен pilot-1: 89 workers; 42 пары writer/restore совпали полностью. Duration
0/1/2/5/10 на 8000–8003: после 1 все A/B final 24/24; 0 и 10 полные состояния
совпали с 05. P-LL из одного состояния: OFF 12 units, magnitude/random/highest 6.
Magnitude сохранил A/B/C 40/40, random потерял один класс каждого seed; highest
тоже 40/40, поэтому преимущество именно low-utility ranking не подтверждено.
Стоимость поиска: 432 -> 216 координат/query; медианы query CPU ~1.90–2.03x.
Старые lake prefixes сохранены. Stages NOT_EVALUATED: delta не имеет контракта
native staged runner; полный этап 06 не объявлен завершённым.

Offline-аудит усилен post-data, по audit-amendment.md; условия обучения неизменны.
Исправлена ошибка нового lake reader: u32 после имени — seq, не длина. Два аудита
совпали, 10 constructed corruptions отвергнуты. Closing verification: 483 blobs
повторно совпали, 4 original lakes и training harness receipts неизменны, repo чистый.
Отчёт 06-pruning.md и три self-review pass сохранены. STATUS оставлен координатору.

Открытые вопросы сведены к границам: совместимый staged learner/механизм и равная
измеренная стоимость; перенос на 05/Brain, powered campaign и production continuity.
Следующее действие: отдельно задать опыт со стадиями на совместимом learner.
