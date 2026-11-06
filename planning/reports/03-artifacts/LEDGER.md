# Этап 03 — рабочий журнал

Цель: проверяемый сравнительный стенд на задачах 02; отдельные train/dev/final, контроли, пересчёт, чувствительность, обоснование будущего N.

Проверено: отчёты 01–02 доступны; clean source HEAD c4c10b285f1741aa798d5ee9c5296ae371fabecf; AGENTS, AGENT_COLD_START, POLICIES P0–P8 и invariants, SECURITY прочитаны. Готовность предыдущих этапов принимается в scope минимального learner; численные результаты проверяются по raw/manifest. Новейший execution record c4c10b28 — исторический upstream, не местная battery. STATUS отстаёт.

Выбрано: A_c1d из существующего генератора; открытые dev seeds 8000–8003; trained/frozen/shuffled; `_snapshot` вместо недостаточного для replay `learner_state`. Исходники не меняются; Git-ветка не требуется для внешнего журнала по соглашению плана.

Выполнено: pilot-1, 14 успешных subprocess; 4 exposed seeds × trained/frozen/shuffled + checks + повтор. Trained final 48/48 и stable-10 на каждом seed. Истинно frozen snapshot неизменен. Повтор 8000 совпал побайтно. 18 offline synthetic cases + positive и 9 транспортных/pipeline failures обнаружены. Два окончательных анализа побайтно совпали по summary/sensitivity/table/power. Raw неизменны. Поздние ужесточения reader раскрыты в post-data-amendments.md.

Мощность: paired SD .2204792759, условный normal known-σ fixed N31/cap48 при δ.10; normal/bootstrap null/δ simulations выполнены. Это DRAFT / NOT LAUNCH READY: реальная σ/хвосты и подтверждающий t/robust дизайн не установлены по 4 exposed seeds. Никакой confirmation, научного null или удаления механизма.

Выходы: ../03-evaluation.md, record-format.md, preregistration-template.md, pilot-1/, final-audit-1/ и final-audit-2/, failure-checks.json. STATUS оставлен координатору.

Следующее действие: итоговая read-only проверка артефактов и чистоты checkout; затем передача этапу 04 без его запуска. Открытые внешние зависимости: полноценный Brain/B8/battery, новые назначенные seeds и принятый подтверждающий дизайн; сохранение навыка и задачи A/B относятся к 04–05.
