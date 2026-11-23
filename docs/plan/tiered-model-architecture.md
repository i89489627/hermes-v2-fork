# Tiered Model Architecture: Lightweight Local для Memory Ops

**Design document / спорная гипотеза** — 2026-11-18

> «Что если все описанные операции для работы долговременной памяти будет выполнять
> не основная модель, а легковесная локальная?»

Это отдельный документ для самой спорной гипотезы из всей серии. Сохранён отдельно,
потому что в отличие от предыдущих (atomic graph, GEPA-оптимизация) у этой **реальные
quality risks и hardware requirements**, которые надо трезво оценивать.

---

## 0. Контекст

Документ описывает **tiered model architecture** для memory pipeline харнесса ZCode.
Базовая архитектура (atomic markdown graph в Git, trust cards, extractor+reviewer pipeline,
CBDC-gated proactive injection, GEPA self-optimization) описана в
[session-memory-zcode-plugin.md](./session-memory-zcode-plugin.md). Этот документ
расширяет её гипотезой: **memory operations выполняет lightweight local model, не main
reasoning model**.

Гипотеза спорная, потому что:

- Малые модели имеют quality ceiling, особенно на review/dedup с subtle contradictions
- Нужен GPU/NPU для разумной latency (CPU-only слишком медленный)
- Output format stability хуже (malformed JSON/YAML risk)
- Multilingual projects могут не иметь подходящей локальной модели
- Coupling к Unsloth/Ollama runtime создаёт dependency

Но потенциальные gain-ы **огромные**: cost → $0 для memory ops, latency 50-200ms vs
1-5s (критично для `UserPromptSubmit` 5s timeout), privacy, decoupling от main model,
и漂亮-convergence с Unsloth Desktop.

---

## 1. Какие операции вообще LLM-тяжёлые

Из всей архитектуры LLM-вызовы делают эти 7 операций. Все остальные (BM25 retrieval,
FTS5 search, Git writes, trust anchors от test green/red, parsing) — не LLM вообще:

| Операция | Частота | LLM-нагрузка | Quality bar |
|---|---|---|---|
| **Extraction** (turn → draft facts) | Каждый turn | Средняя | Medium |
| **Review/dedup** (merge drafts в существующие ноты) | Каждый turn | Средняя | Medium-High |
| **CBDC gating** (use/verify/ignore) | Каждый запрос к модели | Низкая | Medium (но false positive дорог) |
| **Query rewriting** (user prompt → retrieval queries) | Каждый запрос | Низкая | Medium |
| **Compaction summarization** | Каждый turn | Средняя | High (quality влияет на retrieval) |
| **Trust scoring** | При verify-вызовах | Низкая | Low (в основном rules) |
| **GEPA reflection** | Offline, weekly | Высокая | **Critical** |

Вывод: **6 из 7 операций — кандидаты на локальную модель**. GEPA reflection — единственная,
где качество критично и локальную модель использовать опасно.

---

## 2. Pros: почему гипотеза может быть правильной

### 2.1. Cost

Memory ops high-frequency: каждый turn → extraction + review + compaction; каждый query
→ gating + rewrite. На cloud API это могло бы стоить $0.001-0.01 за turn → $10-100/месяц
за активное использование. Локальная модель = $0.

### 2.2. Latency

`UserPromptSubmit` hook имеет **5-секундный timeout** в ZCode. Cloud API call к
Claude/GPT = 1-5s плюс network. Локальная модель (Qwen-3B на Apple Silicon) = 50-200ms.
Это разница между «proactive injection работает» и «hook таймаутит и memory не инжектируется».

### 2.3. Privacy

Memory contents (факты о проекте, предпочтения пользователя, prod-секреты) **остаются
локальными**. Не уходят в Anthropic/OpenAI/DeepSeek логи. Особенно важно для trust card
с конфиденциальными anchors.

### 2.4. Offline

Работает в air-gapped средах, в plane mode, при сетевых сбоях. Memory pipeline не падает,
когда cloud API down.

### 2.5. No rate limits

Можно делать brute-force retrieval + reranking по всему memory store без concern о quota.
Например, retrieve top-50 candidates и прогнать каждого через gating LLM — на cloud это
дорого, локально бесплатно.

### 2.6. Predictable

Нет model deprecation, нет surprise API changes, нет pricing changes. Один раз настроили
Qwen-2.5-3B — он работает годами.

### 2.7. Prefix cache friendly

System prompt memory pipeline (правила extraction, rubric gating) можно держать в prefix
cache навсегда, потому что модель локальная и не имеет multi-tenant contention.

### 2.8. Decoupled from main model

Можно свапать Claude → GLM → Gemini без перестройки memory pipeline. Memory ops
абстрагированы от main reasoning.

---

## 3. Cons: реальные риски

### 3.1. Quality ceiling

Qwen-2.5-3B misses тонкости, которые Claude Opus ловит. Особенно:

- Subtle contradictions между existing note и new draft (reviewer может пропустить конфликт)
- Cross-language факты (если проект bilingual)
- Long-context reasoning (когда надо держать в голове много фактов сразу)

### 3.2. Hardware requirements

Нужен GPU/NPU для разумной latency:

| Model size | RAM (Q4) | tok/s на M2 Pro | tok/s на CPU |
|---|---|---|---|
| 1.5B | ~3GB | ~50 | ~15 |
| 3B | ~6GB | ~30 | ~8 |
| 7B | ~14GB | ~25 | ~3 |
| 14B | ~9GB (Q4) | ~15 | ~1 |

На CPU-only машине 3B+ painfully slow → fallback на cloud.

### 3.3. Output format stability

Малые модели хуже следуют strict JSON/YAML. Risk malformed trust cards. Нужен **parser
с retry + structured output constraints** через llama.cpp grammar, outlines, или instructor.

### 3.4. Cold start

5-30s на first load. Решение: держать модель warm в памяти (через Ollama service или
Unsloth background process).

### 3.5. Multilingual

Llama-3.2 слаб на non-English. Qwen-2.5 силён на Chinese, OK на русском. Если проект на
русском/китайском — Qwen preferred.

### 3.6. Maintenance

Нужно выбрать quantization (Q4_K_M vs Q5_K_M vs Q8), следить за updates, при необходимости
re-benchmark.

### 3.7. Distillation overhead

Если хотим, чтобы локальная модель была хорошо под задачу, нужен fine-tuning. Это отдельная
работа — но это же и opportunity (см. раздел 7).

---

## 4. Конкретные модели по операциям

| Операция | Recommended model | Альтернатива | RAM (Q4) |
|---|---|---|---|
| **Extraction** | Qwen-2.5-3B-Instruct | Llama-3.2-3B | ~2GB |
| **Review/dedup** | Qwen-2.5-3B-Instruct | Phi-3-mini-4k | ~2GB |
| **CBDC gating** | Llama-3.2-1B-Instruct | Gemma-2-2B | ~700MB |
| **Query rewriting** | Qwen-2.5-1.5B | Phi-3-mini | ~1GB |
| **Compaction** | Qwen-2.5-7B-Instruct | Hermes-2-Theta-Llama-3-8B | ~4.5GB |
| **Trust scoring** | Rule-based + Llama-3.2-1B для edge cases | — | ~700MB |
| **GEPA reflection** | **НЕ локальная** — cloud medium (Claude Sonnet / GPT-4o-mini) или Qwen-2.5-14B+ локально | — | ~9GB если локально |

**Total RAM для полного локального memory pipeline** (без GEPA): ~5-7GB. С GEPA локально:
~14-16GB. На современной Mac/PC с 16-32GB RAM — comfortably fits.

Критерии выбора:

- **Structured output support** — Qwen-2.5 хорошо следует JSON schema, Llama-3.2 средне
- **Multilingual** — для русского/китайского Qwen preferred
- **Context length** — для compaction нужна ≥8k context (Qwen-2.5-7B: 32k)
- **Quantization availability** — GGUF Q4_K_M широко доступен для всех recommended models
- **Inference engine compatibility** — Ollama, llama.cpp, MLX (Apple Silicon), vLLM

---

## 5. Архитектура с tiered models

```
┌────────────────────────────────────────────────────────┐
│  Tier 3: GEPA reflection (offline, weekly)             │
│  Cloud medium (Claude Sonnet / GPT-4o-mini)            │
│  ИЛИ Qwen-2.5-14B локально если есть RAM               │
│  Quality-critical, не экономить                        │
└───────────────────────▲────────────────────────────────┘
                        │ reflective dataset
                        │
┌───────────────────────┴────────────────────────────────┐
│  Tier 2: Main agent (cloud или heavy local)            │
│  Claude / GLM-5.3 / Gemini / Llama-3.3-70B             │
│  → Actual coding/reasoning work                        │
│  → Получает injected context из Tier 1                 │
└───────────────────────▲────────────────────────────────┘
                        │ additionalContext (verified facts)
                        │
┌───────────────────────┴────────────────────────────────┐
│  Tier 1: Memory pipeline (lightweight local)           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │extractor │ │ reviewer │ │   gate   │ │  query   │  │
│  │ Qwen-3B  │ │ Qwen-3B  │ │Llama-1B  │ │rewriter  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ │ Qwen-1.5B│  │
│       └─────────────┴─────────────┘     └────┬─────┘  │
│                     │                         │        │
│                     ▼                         ▼        │
│       ┌──────────────────────────────────────────────┐ │
│       │ Trace Store (JSONL) + Pareto front (JSON)    │ │
│       │ + Memory graph (Markdown в Git)              │ │
│       └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

Tier 1 → Tier 2: memory pipeline инжектирует `additionalContext` через `UserPromptSubmit`
hook. Main agent не знает, что факты прошли через Qwen-3B, ему всё равно.

Tier 2 → Tier 3: traces из main agent + memory pipeline собираются в Trace Store,
периодически (раз в неделю) отдаются GEPA optimizer на cloud medium для reflection.

Tier 3 → Tier 1: GEPA возвращает evolved prompts, которые deploy-ятся в memory pipeline.
Pareto front хранит лучшие варианты per (component × model variant).

---

## 6. Реализация через ZCode plugin

Тут есть **beautiful convergence** с тем, для чего Unsloth Desktop создавался: `unsloth
start claude` подключает Claude Code к локальной модели. В нашем случае — main agent
работает через cloud, а memory pipeline через Unsloth-managed local models.

### 6.1. Вариант A: Subagents с model override

Каждый memory subagent явно указывает локальную модель в frontmatter:

```yaml
# agents/memory-extractor.md
---
name: memory-extractor
description: Extract draft facts from conversation turns
model: qwen2.5-3b-instruct
runtime: unsloth
---
You are a fact extractor. Read the conversation turn and extract...
```

ZCode через Unsloth Desktop поднимает локальный Qwen-3B и направляет subagent-вызовы туда.

### 6.2. Вариант B: MCP server как шлюз к локальной модели (preferred)

Plugin экспонирует MCP tools, внутри которых — вызовы к локальной модели через
Ollama/llama.cpp/Unsloth API:

```python
# mcp-server/memory_ops.py
from ollama import Client
from mcp.server import MCPServer, tool

class MemoryOpsServer(MCPServer):
    def __init__(self):
        self.extractor = Client(model='qwen2.5-3b')
        self.reviewer = Client(model='qwen2.5-3b')
        self.gate = Client(model='llama-3.2-1b')
        self.rewriter = Client(model='qwen2.5-1.5b')
        self.compactor = Client(model='qwen2.5-7b')

    @tool
    def extract_facts(self, turn: str) -> list[dict]:
        """Extract draft facts from a conversation turn"""
        response = self.extractor.chat(
            messages=[{"role": "user", "content": extract_prompt(turn)}],
            format='json',  # structured output via llama.cpp grammar
        )
        return parse_facts(response)

    @tool
    def cbdc_gate(self, candidates: list, context: str) -> list[dict]:
        """Decide use/verify/ignore for each retrieved candidate"""
        response = self.gate.chat(
            messages=[{"role": "user", "content": gate_prompt(candidates, context)}],
            format='json',
        )
        return parse_gating(response)

    @tool
    def rewrite_query(self, user_prompt: str) -> list[str]:
        """Rewrite user prompt into multiple retrieval queries"""
        ...
```

Main agent (Claude/GLM) вызывает `mcp__memory-graph__extract_facts` как обычный tool —
не знает и не волнуется, что внутри локальная модель.

**Вариант B предпочтительнее**, потому что:

- Полностью развязывает main agent от memory pipeline
- Позволяет менять локальные модели без изменения main agent behavior
- Даёт unified tracing (через MCP tool call logs)
- Поддерживает graceful fallback (если local недоступен — fallback на cloud tool)
- Позволяет per-operation model selection через конфиг

### 6.3. Конфиг для per-operation model selection

Расширяем `plugin.json` из базовой архитектуры:

```json
{
  "name": "zcode-memory-graph",
  "userConfig": {
    "models": {
      "extractor": {
        "type": "string",
        "default": "qwen2.5-3b-instruct",
        "description": "Model for fact extraction"
      },
      "reviewer": {
        "type": "string",
        "default": "qwen2.5-3b-instruct",
        "description": "Model for review/dedup"
      },
      "gate": {
        "type": "string",
        "default": "llama-3.2-1b-instruct",
        "description": "Model for CBDC gating"
      },
      "rewriter": {
        "type": "string",
        "default": "qwen2.5-1.5b",
        "description": "Model for query rewriting"
      },
      "compactor": {
        "type": "string",
        "default": "qwen2.5-7b-instruct",
        "description": "Model for compaction summarization"
      },
      "reflector": {
        "type": "string",
        "default": "claude-sonnet",
        "description": "Model for GEPA reflection — cloud or large local"
      }
    },
    "runtime": {
      "type": "string",
      "enum": ["ollama", "llama-cpp", "unsloth", "vllm", "cloud-fallback"],
      "default": "ollama"
    },
    "fallback_to_cloud": {
      "type": "boolean",
      "default": true,
      "description": "Fall back to cloud model if local unavailable"
    }
  }
}
```

### 6.4. Graceful fallback

Если local model недоступна (Ollama не запущен, нет RAM, нет GPU), memory ops должны
fallback на cloud через тот же MCP interface. Это критично для UX — memory pipeline
не должен молча ломаться.

```python
@tool
def extract_facts(self, turn: str) -> list[dict]:
    try:
        return self._call_local(self.extractor, turn)
    except LocalModelUnavailable:
        if self.config.fallback_to_cloud:
            return self._call_cloud("claude-sonnet", turn)
        raise
```

Trust card должен отмечать, какая модель использовалась:

```yaml
---
fact: "Project uses pnpm for package management"
trust_score: 0.7
verified_by: "local:qwen-3b:v2.1"
verified_at: 2026-11-18T14:32:00Z
fallback_used: false
---
```

Если fallback сработал, `verified_by` становится `cloud:claude-sonnet-4` и trust_score
повышается (cloud model надёжнее).

---

## 7. Distillation angle — третий слой улучшения

GEPA оптимизирует промпты. Distillation улучшает саму локальную модель под ваш домен.
Это **orthogonal improvement**, который со временем закрывает quality gap между local
и main model.

Цепочка:

1. **Phase A: Cold start** — main model (Claude) делает extraction в early phase.
   Traces копятся в Trace Store с тегом `model: claude-opus-4`.

2. **Phase B: GEPA reflective dataset** — из traces формируется reflective dataset:
   для каждого неудачного случая (turn, drafts, what_went_wrong).

3. **Phase C: Distillation** — эти traces используются для fine-tuning Qwen-3B через
   Unsloth Desktop. LoRA на 1-2 epochs достаточно для domain adaptation.

4. **Phase D: Switch** — после distillation локальная модель учится imitate качество
   main model на вашем домене. Переключаем extraction на local.

5. **Phase E: Continuous** — main model всё ещё используется для actual reasoning work
   и для GEPA reflection. Local для high-frequency memory ops.

По сути: **сначала main model учит локальную, потом локальная делает работу сама, а
main model используется только для reflection и actual reasoning work.**

Это exactly то, для чего Unsloth Desktop и создавался — fine-tune small models для
конкретного use case. В архитектуре это становится **четвёртым применением Unsloth**:

1. Unsloth как runtime для subagent-ов (вариант A)
2. Unsloth как inference backend для MCP server (вариант B)
3. Unsloth как fine-tuning engine для periodic distillation
4. Unsloth для quantized deployment (GGUF Q4_K_M для production)

### 7.1. Distillation dataset schema

```json
{
  "turn_id": "abc123",
  "user_prompt": "Add a /health endpoint to the API",
  "model_used": "claude-opus-4",
  "extracted_facts": [
    {
      "type": "Project",
      "text": "API needs /health endpoint",
      "confidence": 0.95
    }
  ],
  "review_outcome": "added",
  "reviewer_model": "claude-opus-4",
  "gepa_feedback": "Good extraction, but missed that this was discussed before",
  "suitable_for_distillation": true,
  "distillation_split": "train"
}
```

### 7.2. LoRA config для distillation

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen/qwen2.5-3b-instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=load_distillation_dataset(),
    formatting_func=format_as_instruction_pairs,
    max_seq_length=4096,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
)
trainer.train()
```

После training — merge LoRA в base model, quantize в GGUF Q4_K_M, deploy в Ollama.

---

## 8. Когда НЕ использовать локальную модель

### 8.1. GEPA reflection

Quality-critical, дорогие ошибки. Лучше cloud medium (Claude Sonnet / GPT-4o-mini) или
14B+ локально. GEPA paper специально подчёркивает, что reflection model quality —
non-negotiable.

### 8.2. Initial extraction в cold-start проекте

Пока memory pipeline не имеет traces, локальная модель не знает вашего домена. Сначала
main model + traces (Phase A из distillation chain), потом distillation (Phase C), потом
переключение на локальную (Phase D). Это 2-4 недели реального использования.

### 8.3. Compaction summarization для critical facts

Если факт про prod-config или security, лучше main model. Локальная может пропустить
нюанс. Trust card должен иметь `critically: high` flag, и при compaction high-criticality
фактов использовать main model автоматически.

### 8.4. Multilingual project без подходящей локальной модели

Если проект на иврите или хинди, и доступные small models плохо на нём работают, лучше
cloud. Qwen-2.5 силён на Chinese/English, OK на русском/испанском. Llama-3.2 слаб на
non-Latin scripts.

### 8.5. CPU-only machine без GPU/NPU

3B+ models painfully slow на CPU (~8 tok/s). Это даёт 30-60s latency на extraction,
что превосходит `UserPromptSubmit` timeout. Решение: либо cloud fallback, либо 1B model
на CPU (~15 tok/s, может уложиться).

### 8.6. Single-user setup без distillation budget

Если в команде 1 пользователь и нет ресурсов на fine-tuning, локальная модель не получит
domain adaptation. Cloud API с good prompt может быть качественнее, чем generic Qwen-3B.
Distillation имеет смысл при ≥10 hours of traces.

---

## 9. Конфликты с базовой архитектурой

### 9.1. GEPA metrics переопределяются

GEPA optimizes prompts против metric. Если локальная модель слабее, prompt optimisation
может overfit под её weaknesses. Например, GEPA может научить extractor prompt избегать
complex syntactic structures, потому что Qwen-3B на них путается — но это снижает
expressiveness.

Решение: GEPA evaluate должен использовать **ту же модель, что и в проде** (т.е. локальную),
а не main model. Pareto front per (component × model variant).

### 9.2. Trace Store становится training data

Раньше traces были только для GEPA reflection. Теперь они ещё и training set для
distillation. Нужна versioning:

```
${CLAUDE_PLUGIN_DATA}/traces/
├── v1.0/                    # initial traces, main model
│   ├── 2026-11-18.jsonl
│   └── 2026-11-19.jsonl
├── v1.1/                    # after switching extractor to local Qwen-3B
│   ├── 2026-12-01.jsonl
│   └── ...
├── distillation-ready/
│   └── train.jsonl          # filtered for distillation quality
└── pareto-front.json
```

Каждый trace имеет:

```json
{
  "turn_id": "...",
  "models_used": {
    "extractor": "claude-opus-4",
    "reviewer": "qwen2.5-3b",
    "gate": "llama-3.2-1b"
  },
  "suitable_for_distillation": true,
  "distillation_target": "extractor",  # or null
  "quality_label": "good"  # good | acceptable | bad
}
```

### 9.3. Pareto front per model variant

Если у вас несколько локальных моделей в rotation (Qwen-3B для extraction, Llama-1B для
gating), GEPA должен оптимизировать prompts per model. Pareto front расширяется:

```json
{
  "extractor": {
    "qwen2.5-3b": [...candidates...],
    "llama-3.2-3b": [...candidates...]
  },
  "gate": {
    "llama-3.2-1b": [...candidates...],
    "gemma-2-2b": [...candidates...]
  }
}
```

Routing layer (через `PreToolUse` hook) выбирает, какой model variant использовать,
потом — какой candidate из Pareto front этой модели.

### 9.4. Trust card provenance extension

`verified_by` теперь может быть `local:qwen-3b:v2.1` вместо `claude-opus-4.1`. Trust
score должен учитывать capability model:

| verified_by pattern | Trust weight |
|---|---|
| `test:green` (rule-based anchor) | 1.0 |
| `code-review:human` | 0.95 |
| `cloud:claude-sonnet-4` | 0.85 |
| `cloud:gpt-4o` | 0.85 |
| `local:qwen-14b` | 0.7 |
| `local:qwen-3b` | 0.55 |
| `local:llama-1b` | 0.4 |
| `local:llama-1b:fallback` | 0.35 |
| `llm-only` (без anchor) | 0.3 |

Trust score вычисляется как weighted average of all verification events.

### 9.5. Cold start UX

Первый запуск плагина: main model может работать, но memory pipeline молчит 5-30s пока
грузится Qwen. Решения:

- Lazy load с background warmup при `SessionStart` hook
- Splash indicator «Memory pipeline loading…»
- Fallback на main model если local недоступен в течение 10s

---

## 10. Спорные места гипотезы

Эти пункты надо отдельно продумать перед реализацией.

### 10.1. Latency на CPU-only машинах

Если у пользователя нет GPU/NPU, 3B model даёт ~8 tok/s. Extraction prompt ~500 tokens
output → 60 seconds. Это превосходит любой разумный timeout.

**Options:**

- Mandate GPU/NPU как системное требование (отрезать часть пользователей)
- Использовать 1B model на CPU (~15 tok/s, ~30s) с degraded quality
- Fallback на cloud автоматически при CPU-only detection
- Async extraction — extract в background, не блокируя turn

### 10.2. Output format stability

Малые модели часто ломают JSON schema, особенно при сложной структуре (nested arrays,
optional fields). Это создаёт parser errors и retries, которые удваивают latency.

**Mitigations:**

- llama.cpp grammar constraints — guaranteed valid JSON
- outlines / instructor — Python libraries для structured output
- Retry with simplified prompt (3 attempts max)
- Fallback на main model после 3 failed retries
- Schema validation с подробным error feedback на retry

### 10.3. Quality gap на review/dedup

Это **самое слабое место**. Review/dedup требует понимать, является ли new draft
semantic duplicate of existing note, и может ли он быть merged. Это требует reasoning
качества, которого 3B models часто не имеют.

**Mitigations:**

- Использовать 7B+ model для review/dedup (Qwen-2.5-7B-Instruct)
- Distillation с main model traces (после 2-4 недель использования)
- Conservative policy: если reviewer не уверен → не merge, а add как new note с
  `pending_review: true` flag
- Periodic main-model cleanup: раз в неделю main model проходит через pending_review
  и решает, что merge

### 10.4. Coupling к Unsloth/Ollama runtime

Plugin становится зависимым от external runtime (Ollama service, Unsloth Desktop).
Если Ollama не установлен — plugin не работает.

**Mitigations:**

- Plugin bundling: включить minimal llama.cpp binary в plugin (увеличит размер на ~50MB)
- Auto-install Ollama через plugin install script
- Graceful fallback на cloud если runtime недоступен
- Чёткое документирование системных требований

### 10.5. Distillation feedback loop может ухудшать quality

Если distillation dataset contaminated (включает bad extractions от main model, которые
прошли review с ошибками), локальная модель учится повторять ошибки.

**Mitigations:**

- Quality filtering перед distillation: только traces с `quality_label: good`
- GEPA reflection перед distillation: GEPA оценивает traces и фильтрует bad ones
- Human review на sample из distillation dataset (раз в месяц)
- Distillation versioning: сохранять snapshot каждой версии, можно rollback если
  новая версия хуже

### 10.6. Cost inversion для маленьких проектов

Если проект маленький и пользователь делает мало запросов в день (<20), cloud API
стоит <$1/месяц. Local model + Ollama + Unsloth + periodic distillation стоит больше
в engineer time и hardware.

**Mitigation:**

- Document break-even: « имеет смысл если >100 turns/day или privacy requirement»
- Hybrid mode: extraction на cloud до достижения N traces, потом switch на local
  после distillation

---

## 11. Обновлённый фазовый план

| Фаза | Что строить | Сложность |
|---|---|---|
| MVP | Stop hook → extractor (main model) → MEMORY.md, BM25 retrieval | Неделя |
| v0.2 | Reviewer subagent + trust card frontmatter (main model) | +Неделя |
| v0.3 | Ground-truth anchors через PostToolUse | +Неделя |
| v0.4 | CBDC gating, hybrid retrieval, prefix cache preservation | +2 недели |
| **v0.4.5** | **MCP server для memory ops, model-agnostic interface** | **+Неделя** |
| **v0.5** | **Trace Store + метрики; migrate extractor на local Qwen-3B** | **+2 недели** |
| **v0.5.5** | **Migrate gating на Llama-1B, query rewriter на Qwen-1.5B** | **+Неделя** |
| **v0.6** | **GEPA adapter для extractor (local), Pareto front per local model variant** | **+2 недели** |
| **v0.6.5** | **Compaction на local Qwen-7B (с fallback на main для critical facts)** | **+Неделя** |
| v0.7 | GEPA для gating и query rewriter (local), Pareto routing | +2 недели |
| **v0.7.5** | **Distillation loop: main model traces → Unsloth fine-tune local models** | **+2 недели** |
| v0.8 | GEPA для trust rubric и compaction, full self-improving loop | +2 недели |
| v1.0 | Git sync, concurrent write locking, role separation | +2 недели |

---

## 12. Итог

Гипотеза **архитектурно sound и aligns с industry direction**, но **спорная** —
нужно трезво оценивать quality risks и hardware requirements перед реализацией.

Главные gain-ы:

- **Cost → $0** для memory ops
- **Latency 50-200ms vs 1-5s** (критично для `UserPromptSubmit` 5s timeout)
- **Privacy** — memory contents остаются локальными
- **Decoupling** от main model, можно свапать cloud providers
- **Beautiful convergence с Unsloth Desktop** — runtime + distillation engine

Главные риски:

- **Quality gap на review/dedup** (самое слабое место, решается distillation)
- **Hardware requirements** (нужен GPU/NPU, fallback на cloud если нет)
- **Output format stability** (решается llama.cpp grammar / outlines / instructor)
- **GEPA reflection — НЕ переносить на локальную**, quality-critical
- **Distillation feedback loop** может ухудшать quality (решается quality filtering)

Архитектурно это **layering, не replacement**:

- **Tier 1**: Local lightweight (extractor, gate, rewriter, compaction) — high-frequency
- **Tier 2**: Main cloud/heavy (actual coding work)
- **Tier 3**: Cloud medium или local 14B+ (GEPA reflection, distillation)

Третий layer improvement поверх GEPA: **distillation loop**, где main model сначала
учит локальную на своих traces, потом локальная работает сама. Это exactly то, для чего
Unsloth Desktop создавался.

**Рекомендация**: реализовать как **v0.4.5-v0.7.5 phases** в общем roadmap, после того
как базовая архитектура (atomic graph, trust cards, CBDC gating) уже работает на main
model. Не пытаться делать local-first с самого начала — нужно накопить traces для
distillation. Использовать MCP server pattern (вариант B) для развязки main agent от
memory pipeline. Graceful fallback на cloud обязателен.

---

## 13. Ссылки

- Base architecture: [session-memory-zcode-plugin.md](./session-memory-zcode-plugin.md)
- ZCode memory: https://zcode.z.ai/en/docs/memory
- ZCode plugin docs: https://zcode.z.ai/en/docs/plugin
- ZCode hooks: https://zcode.z.ai/en/docs/hooks
- Unsloth Desktop: https://unsloth.ai
- Unsloth docs: https://unsloth.ai/docs
- Unsloth Start (connect agents to local models): https://unsloth.ai/docs
- Ollama: https://ollama.com
- llama.cpp grammar constraints: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
- outlines (structured output): https://github.com/outlines-dev/outlines
- instructor (structured output): https://github.com/jxnl/instructor
- Qwen-2.5 models: https://qwenlm.github.io/blog/qwen2.5
- Llama-3.2 models: https://ai.meta.com/llama
- GEPA: https://github.com/gepa-ai/gepa
- GEPA paper: https://arxiv.org/abs/2507.19457
