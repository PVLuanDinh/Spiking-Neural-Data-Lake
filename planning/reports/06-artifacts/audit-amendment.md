# Дополнение аудита после начала pilot-1

2 ноября 2026 года. После начала блока усилен только offline-reader analyze.py:
добавлены независимый расчёт ответов из beta, разбор named-state frames самого
lake, проверка сохранения старого byte-prefix и четыре неисправности duration.
Ранее этот reader проверял ответы по scorer и рестарту, но не разбирал lake.
Замороженная исходная версия лежит в pilot-1/provenance. Training harness,
протокол, seeds, бюджеты, masks и данные не меняются; повторного обучения ради
аудит-изменения нет. Это post-data verification amendment, не preregistration.

Первый positive разбор реального lake выявил ошибку нового reader: u32 после
имени был ошибочно прочитан как длина blob. В home.py::_parse_state_record это
sequence. Reader исправлен по фактическому формату до итогового аудита; training
и сохранённые lake не затронуты. Это ошибка проверки, не повреждение памяти.
