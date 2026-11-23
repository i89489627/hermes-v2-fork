"""Bundled skills index (plan P21.T4, issue #156140).

Importable python module rather than a data file so the index ships inside
the package without pyproject package-data wiring (documented deviation in
:mod:`hermes.skills.index`): ~30 realistic skill records covering the
categories the fork actually uses, served instantly on cold cache — the fix
for the 86-second cold search (52MB download) reported in issue #156140.
"""

from __future__ import annotations

#: The bundled base skills index: ``name``, ``description``, ``category``,
#: ``version`` per record (kept in sync with the fork's skill set).
INDEX: list[dict[str, str]] = [
    {"name": "commit-message", "description": "Write conventional, scoped commit messages from staged diffs", "category": "coding", "version": "1.4.0"},
    {"name": "pr-review", "description": "Review a pull request for correctness, tests and style drift", "category": "coding", "version": "2.0.1"},
    {"name": "release-notes", "description": "Draft release notes from merged commits and changelog entries", "category": "docs", "version": "1.1.3"},
    {"name": "changelog-summarizer", "description": "Summarize keep-a-changelog entries into user-facing highlights", "category": "docs", "version": "1.0.2"},
    {"name": "sql-explainer", "description": "Explain a SQL query's plan and suggest index improvements", "category": "data", "version": "1.2.0"},
    {"name": "csv-cleaner", "description": "Normalize, deduplicate and type-cast messy CSV files", "category": "data", "version": "1.3.1"},
    {"name": "pdf-extract", "description": "Extract text and tables from PDF documents for downstream tools", "category": "data", "version": "1.0.0"},
    {"name": "regex-builder", "description": "Build and test regular expressions from natural-language examples", "category": "coding", "version": "1.5.2"},
    {"name": "cron-schedule", "description": "Compose and validate cron expressions with timezone handling", "category": "devops", "version": "1.1.0"},
    {"name": "env-audit", "description": "Audit environment variables for secrets, conflicts and drift", "category": "security", "version": "2.1.0"},
    {"name": "secret-scan", "description": "Scan a tree for leaked credentials before pushing", "category": "security", "version": "2.0.0"},
    {"name": "deps-audit", "description": "Check dependency manifests for known CVEs and version drift", "category": "security", "version": "1.7.4"},
    {"name": "log-triage", "description": "Cluster error logs and surface the most probable root cause", "category": "devops", "version": "1.2.2"},
    {"name": "docker-compose-debug", "description": "Diagnose docker-compose service failures and port conflicts", "category": "devops", "version": "1.3.0"},
    {"name": "k8s-rollout", "description": "Drive and verify Kubernetes rollout status and rollback steps", "category": "devops", "version": "1.0.4"},
    {"name": "terraform-plan-review", "description": "Review terraform plans for destructive or unexpected changes", "category": "devops", "version": "1.1.1"},
    {"name": "git-bisect-helper", "description": "Automate a git bisect session with a test command", "category": "coding", "version": "1.0.3"},
    {"name": "unit-test-scaffold", "description": "Generate pytest scaffolding with markers and fixtures for a module", "category": "coding", "version": "2.2.0"},
    {"name": "test-data-gen", "description": "Generate deterministic synthetic test fixtures and edge cases", "category": "coding", "version": "1.2.1"},
    {"name": "eval-set-builder", "description": "Assemble labeled eval sets from memory entries and transcripts", "category": "memory", "version": "1.0.2"},
    {"name": "memory-defrag", "description": "Merge duplicate memory entries and refresh stale trust cards", "category": "memory", "version": "1.1.0"},
    {"name": "trust-card-audit", "description": "Audit trust-card verification events for missing provenance", "category": "memory", "version": "1.0.1"},
    {"name": "prompt-lint", "description": "Lint prompt templates for cache-safety and role alternation", "category": "coding", "version": "1.4.1"},
    {"name": "api-mocker", "description": "Stand up a local mock HTTP API from an OpenAPI spec", "category": "web", "version": "1.2.0"},
    {"name": "screenshot-diff", "description": "Compare UI screenshots and highlight visual regressions", "category": "web", "version": "1.0.5"},
    {"name": "browser-replay", "description": "Replay a recorded browser session against a staging environment", "category": "web", "version": "0.9.2"},
    {"name": "email-draft", "description": "Draft concise email replies from thread context and tone hints", "category": "communication", "version": "1.3.3"},
    {"name": "meeting-notes", "description": "Turn raw meeting transcripts into decisions and action items", "category": "communication", "version": "1.1.2"},
    {"name": "slack-digest", "description": "Summarize a busy Slack channel into a daily digest", "category": "communication", "version": "1.0.0"},
    {"name": "timezone-converter", "description": "Convert meeting times across timezones with DST awareness", "category": "communication", "version": "1.0.7"},
]
