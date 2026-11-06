# Этап 04 — журнал работы

Цель: воспроизводимый baseline readout поставленной задачи на фиксированном резервуаре,
с R-STDP/no-tune/shuffle, delta и carry/reset; передача ограниченного результата этапу 05.

Проверено: отчёты 02–03 готовы в scope source learner; checkout чистый на c4c10b28.
Прочитаны AGENTS, cold start, SECURITY, POLICIES P0–P8/invariants и нужные компоненты G5.
T121/L1b-STALE учтён, его раздел результата пуст; shared code не меняется.
Подготовлена отдельная .venv-04: CPython 3.14.6t, numpy 2.5.4, torch 2.15.0; GIL off
после импортов. Первая попытка установки в sandbox не имела DNS; разрешённая
сетевая загрузка в локальный venv завершилась. Учебная .venv-01 не изменена.

Завершено: offline audit этапа 03 совпал; pilot-1 честно остановлен на missing
threadpoolctl; пакет 3.7.0 установлен в .venv-04. pilot-2 COMPLETE: 36 workers,
144 основные записи, полный повтор 8000 совпал. Delta устойчиво решает S0C0,
S1C0/carry и A_c1d на 8000–8003; frozen/shuffle не решают устойчиво. S1C0/reset
имеет точный information ceiling 50%. R-STDP нестабилен, source не ремонтировался.
4 runtime guard faults, 10 analysis faults и 7 transport faults обнаружены.
Два final offline audit совпали побайтно, analysis-source сохранён; отчёт
04-readout.md записан. Никакого powered научного вывода/NULL и замены upstream нет.

Открыто для будущих этапов: пара A/B с совместимыми целями; delta carrier/lake
сохранение и рестарт; конкретный механизм сна; return integration; ограниченный
ремонт Part D R-STDP. Ни один из этих пунктов не выдан за готовый.

Финальная сверка выполнена: 405 source receipts совпали, checkout чистый, ссылки
отчёта существуют, пять файлов двух offline audit побайтно равны.
Следующее действие: передать Артуру файл 04-readout.md. Этап 04 завершён в
объявленном scope. STATUS остаётся координатору; сообщения не отправляются.
