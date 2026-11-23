# Hermes v2.1 — Проверка 7 (повторный dry-run после применения r7)

**Дата:** 2026-11-20
**Версия плана:** v2.1-r7 (применены все 7 пунктов вердикта check 6 — dry-run перепроверяет исполнимость)
**Проверялось:** исполнимость `hermes-v2-tasks-r7.yaml` (112 задач, 21 фаза) как инструкции для coding-агента
**Среда:** изолированный venv — Python 3.12.16, pyyaml 6.0.3, pytest 9.2.0; скретч-репо `.dryrun/repo` (вне deliverables); fork-base клонирован read-only
**Снапшот для сверок:** тег `v2026.9.24` из `NousResearch/hermes-agent` — совпадает с meta.fork_base_version (pin r7 проверен через ls-remote, находка E-PIN)
**Запрос пользователя:** «Примени 7 пунктов из r7 с валидатором. В конце работы не выводи мне кучу файлов на превью, выводи только те что изменены»

**Валидатор r7 (scripts/validate_r7.py):** **133 инвариантов зелёные**

---

## 1. Метод

Проверяется **исполнимость** плана: тот же симулятор харнесса, что в check 6, прогнал 112 задач
в порядке зависимостей и выполнил все машиноисполнимые действия. Симулятор не менялся (кроме stub-а
pyproject.toml, отражающего закреплённый r7 контент) — разница с check 6 показывает, где план стал
исполнимым.
Правки реального кода Hermes не производятся — всё синтезируется в скретч-репо; fork-base
клонируется read-only (v2026.9.24 = pin из meta) для сверки допущений «файл уже существует» с фактом.

| Контур | Что делается |
|---|---|
| A. Граф | Топологическая сортировка Кана → волны исполнения; прямые ссылки; дубль-рёбра; сверка `[parallel]`-аннотаций раздела 7 с графом |
| B. Файлы | Проход задач по волнам: `files_to_create` синтезируется (stub по расширению), конфликты создания, `files_to_modify` — порядок относительно создателя + сверка с реальным деревом fork-base |
| C. Тесты | `tests_to_write` парсится в pytest node-id, синтезируются stub-модули, `pytest --collect-only` обязан найти каждый распознаваемый spec |
| D. Синтакс-гейт | Каждый синтезированный артефакт валидируется: `ast.parse` / `yaml` / `json` / `tomllib` / `bash -n` / непустота |

---

## 2. Контур A — граф зависимостей: 12 волн, максимальная ширина 23

Все 171 рёбер уважены при планировании; незапланированных задач: 0.

| Волна | Задач | Фазы |
|---|---|---|
| 1 | 1 | P0:1 |
| 2 | 2 | P0:2 |
| 3 | 17 | P0:1 P1:2 P10:2 P11:1 P13:1 P19:1 P21:9 |
| 4 | 8 | P1:1 P10:1 P13:1 P14:1 P18:1 P19:1 P20:1 P3:1 |
| 5 | 20 | P1:3 P10:1 P13:3 P14:2 P15:1 P17:2 P18:1 P19:1 P2:1 P21:1 P3:1 P5:1 P8:2 |
| 6 | 23 | P10:1 P13:1 P14:2 P15:3 P17:2 P18:1 P19:1 P2:3 P3:2 P4:1 P5:3 P6:1 P8:2 |
| 7 | 10 | P10:1 P14:2 P16:1 P18:1 P2:1 P3:1 P4:1 P6:1 P8:1 |
| 8 | 8 | P14:2 P20:1 P3:1 P4:1 P6:1 P7:1 P8:1 |
| 9 | 10 | P14:1 P16:2 P20:2 P21:2 P4:1 P7:2 |
| 10 | 7 | P11:1 P16:1 P20:2 P22:1 P7:2 |
| 11 | 4 | P11:1 P16:1 P22:2 |
| 12 | 2 | P11:1 P22:1 |

- Волна 3 подтверждает решение r4: 9 задач Phase 21 (bug-fixes) стартуют сразу после Phase 0.

- Межфазовые зависимости (dep объявлен позже задачи): **10** — это норма для трековой структуры; из них задокументированы в флагах раздела 7: **10**, недокументированы: **0** (WARN — добавить примечание в раздел 7)
- Дубли рёбер: **0**
- Фаз с внутренними рёбрами: 20 (P0:4, P1:5, P2:4, P3:7, P4:3, P5:3, P6:2, P7:3, P8:3, P10:6, P11:3, P13:5, P14:15, P15:2, P16:2, P18:3, P19:4, P20:1, P21:1, P22:4)
- Фазы с аннотацией `[parallel]` в разделе 7: 8 (P5, P6, P13, P15, P17, P18, P19, P21)
- Фазы с внутренними рёбрами БЕЗ аннотации parallel (нормально для последовательных фаз): P0, P1, P2, P3, P4, P7, P8, P10, P11, P14, P16, P20, P22

---

## 3. Контур B — файловый план (скретч-репо синтезирован)

- Уникальных файлов синтезировано из `files_to_create`: **181** (+142 тестовых из `tests_to_write`)
- Файлов, создаваемых более чем одной задачей: **0**
- `files_to_modify` без создателя (сверка с деревом снапшота `v2026.9.24`): **1**, из них реально существуют в upstream: **1**, отсутствуют: **0**
- Изменений раньше создания / в той же волне: **0**
- Изменений без ребра зависимости от создателя: **0**

Сверка допущений fork-base с фактом (снапшот `v2026.9.24`; топ-15 по числу правящих задач):

| Файл | Задач правят | Есть в снапшоте? |
|---|---|---|
| `README.md` | 1 | да |

---

## 4. Контур C — тестовый план и pytest

- Распознано pytest node-id: **409** в 142 файлах
- Whole-file specs (файл без `::теста`): **0**
- Prose-specs (не machine-resolvable): **0**
- Файлов вне дефолтного discovery pytest (`test_*.py`): **0**
- Дублей теста в одном файле между задачами: **0**

`pytest --collect-only` (default import mode (tests/**/__init__.py скелет от P0.T3 — план работает без флагов)): собрано **409**, ожидалось **409**, не найдено **0**, лишних **0**.

---

## 5. Контур D — синтакс-гейт: 307 артефактов, ошибок 0

---

## 6. Находки: 13 (ERROR 0 / WARN 0)

| № | Серьёзность | Код | Находка |
|---|---|---|---|
| F1 | INFO | A-FWD | cross-phase dependency P10.T3 -> P11.T1 (declared later in YAML; documented in section 7 flags) |
| F2 | INFO | A-FWD | cross-phase dependency P10.T5 -> P11.T1 (declared later in YAML; documented in section 7 flags) |
| F3 | INFO | A-FWD | cross-phase dependency P14.T1 -> P18.T1 (declared later in YAML; documented in section 7 flags) |
| F4 | INFO | A-FWD | cross-phase dependency P20.T6 -> P21.T1 (declared later in YAML; documented in section 7 flags) |
| F5 | INFO | A-FWD | cross-phase dependency P20.T6 -> P21.T11 (declared later in YAML; documented in section 7 flags) |
| F6 | INFO | A-FWD | cross-phase dependency P20.T6 -> P21.T2 (declared later in YAML; documented in section 7 flags) |
| F7 | INFO | A-FWD | cross-phase dependency P20.T6 -> P21.T5 (declared later in YAML; documented in section 7 flags) |
| F8 | INFO | A-FWD | cross-phase dependency P21.T6 -> P21.T12 (declared later in YAML; documented in section 7 flags) |
| F9 | INFO | A-FWD | cross-phase dependency P3.T6 -> P11.T1 (declared later in YAML; documented in section 7 flags) |
| F10 | INFO | A-FWD | cross-phase dependency P4.T3 -> P5.T1 (declared later in YAML; documented in section 7 flags) |
| F11 | INFO | A-PAR | [parallel] in section 7 means track-level parallelism vs the backbone; internal edges inside parallel-annotated phases (20 edges across 8 phases) are legal and honored by wave ordering |
| F12 | INFO | E-LAYOUT | fork-base snapshot v2026.9.24: python package is hermes_cli/ + hermes_platform/ + agent/ + gateway/ (NO top-level hermes/ package — 'hermes' is a CLI shim script); plan v2.0 module tree hermes/** is a NEW package; modify-without-creator paths: 1, absent in snapshot: 0 — after the r7 creator-first fix every modify target has a creator or exists upstream |
| F13 | INFO | E-PIN | pinned fork-base v2026.9.24 resolves upstream: e3dd27ee2d8b011737a4eea8e3eb3d711ab78690	refs/tags/v2026.9.24 |

---

## 7. Вердикт

**Dry-run пройден.** Граф планируем (все задачи попадают в волны), файловый план непротиворечив,
каждый машиноисполнимый тест-spec собирается pytest, синтезированные артефакты парсятся.
WARN-находок: 0 — все снабжены рекомендацией и не блокируют исполнение.

Состав применённой ревизии r7 (все 7 пунктов вердикта check 6):

1. **E-PIN:** fork-base закреплён за v2026.9.24 (commit f97608f; v0.18.2 не существует upstream); артефакты миграции переименованы в fork-нейтральные имена;
2. **B-MODMISSING (22):** creator-first файловый план — 20 путей переведены из files_to_modify в files_to_create (задачи Phase 21 авторят v2-модули, которые чинят; hermes/core/agent.py создаёт P0.T3 по Phase 0 Files; P14.T6 создаёт hermes/gateway/health.py вместо каталога);
3. **B-MODBEFORE (2):** рёбра P18.T3 += P14.T4 и P8.T2 += P3.T4 (гонки записи устранены);
4. **B-NODEP (5):** рёбра P11.T2 += P3.T5 и P20.T6 += {P21.T5, P17.T2, P17.T4, P19.T2};
5. **C-IMPORTMODE:** повторный dry-run поймал, что вариант A вердикта (importmode в pyproject.toml) неисполним на pytest 9.2 — у `--import-mode` нет ini-эквивалента; реализован вариант B самого вердикта: `tests/**/__init__.py` пакет-скелет (20 файлов) создаёт P0.T3, pyproject.toml (P0.T1) несёт pytest-конфиг с маркерами;
6. **C-DUPTEST (3):** дублирующиеся тесты переименованы с суффиксом фазы-владельца (_p20, _p8);
7. **A-FWD (2):** флаги раздела 7 документируют P14.T1→P18.T1 и P3.T6→P11.T1; **B-DIRPATH/B-DUPCREATE:** устранены; + 2 структурных ребра для новых создателей (P19.T4 += P19.T2, P21.T6 += P21.T12). Итого 9 новых рёбер: 162 → 171.

Что dry-run НЕ проверяет (принципиальные границы):

- реальный код Hermes не правится: правки существующих файлов симулируются stub-подстановкой;
- снапшот для сверок — v2026.9.24 = pin из meta (совпадают, см. E-PIN); правки существующих
  файлов fork-base по-прежнему симулируются stub-подстановкой;
- runtime-бюджеты (gate 3.5s, injection <5s) — без реального кода мерить нечего;
- семантику acceptance criteria, зависящую от LLM-поведения.

---

## Review metadata

- Check performed by: Z.ai (GLM), по запросу пользователя
- Метод: повторная симуляция исполнения (4 контура A–D) в venv + инвариант-валидатор scripts/validate_r7.py (все проверки зелёные); находки пронумерованы F1–F13
- Previous checks: check 1 = v2.1 self-review, check 2 = r3.1 sync-аудит, check 3 = r3.2 (мутационное
  тестирование), check 4 = r4 (граф + арифметика), check 5 = r6 (перевод + 69 инвариантов),
  check 6 = r6 dry-run (25 блокирующих находок — все устранены в r7)
- Скрипты: `scripts/dryrun_r7.py`, `scripts/validate_r7.py`, `scripts/apply_r7.py` (запуск: `.dryrun-venv/bin/python scripts/dryrun_r7.py`)

