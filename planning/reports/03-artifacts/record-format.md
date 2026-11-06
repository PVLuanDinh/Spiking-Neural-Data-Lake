# Журнал этапа 03, v1

Один каталог на run-id; существующий каталог не перезаписывается. Это журнал эксперимента, не второй lake и не файл навыка.

- `manifest.json`, `sndl.stage03.run.v1`: source HEAD/dirty, exposed-dev статус, полный список seeds/arms, runtime и платформа, UTC начала, хеши файлов, список subprocess-команд, exit codes, wall/CPU seconds, размеры stdout/stderr, причины исключений и статус RUNNING/COMPLETE/HALT. Каждая завершившаяся попытка записывается сразу; незавершённый manifest не проходит reader. Таймаут и ошибка — записи о неисправности, не accuracy=0.
- `provenance/`: точные байты программ, протокола и ключевых исходников на момент запуска. Полный чистый исходный коммит закреплён HEAD. Поздние исправления offline анализа имеют отдельный `analysis_sha256`; старые raw и сводки сохраняются.
- `<arm>-<seed>.json`, `sndl.stage03.arm.v1`: задача/seed/arm, конфигурация, контракт train/dev/final, каждое train-предъявление с истинной и реально предъявленной меткой, начальный и конечный `_snapshot` с replay/cache, все dev/final-ответы и state hashes до/после каждого чтения, checkpoint до final, oracle, данные штатного attempt. Порядок: before_dev → epoch_dev × N → final, ровно один final. SNDL labels в данные reader не передаются.
- `<key>.stderr.txt`: stderr даже при ошибке; непустой stderr запрещает успешное чтение. `trained-8000-repeat.json` — независимое повторное обучение с нуля, не восстановление навыка и не новый seed.
- Отдельная папка анализа: `summary.json`, `table.md`, `sensitivity.json`, `power.json`. Первые три строятся из raw и архивного manifest, без SNDL; power зависит от парных per-seed разностей и хеша summary. Никакого обучения при пересчёте.

Главные поля per-seed таблицы: before_dev_accuracy, last_dev_accuracy, final_correct / final_presentations, stable10 / ever_stable10, epochs, train_presentations, updates. Accuracy — доля правильных предъявлений; повторение одного текста четыре раза не делает четыре независимых наблюдения. Для мощности единица N — seed, не вопрос/эпоха/повтор процесса. `primary_sd` — sample SD парных final-разностей trained − shuffled.

Обязательный reader не принимает неполные arms/seeds, дубль, несогласованные цели/порядок/бюджеты, пересечение splits, изменение snapshot/cache или сломанный контроль. Невалидные входы дают INVALID / NO SCIENTIFIC READING. Успешный reader этого пилота не выдаёт научный PASS или NULL.
