# Multi-Tenant Customer Support & Returns Agent

A production-grade, multi-tenant AI agent service for e-commerce customer support and return management. Built
incrementally from a hardened LLM router up to a governed, observable, enterprise-integrated deployment on GCP.

---

## Architecture Overview

- **API & Routing:** FastAPI with Server-Sent Events (SSE), unified LLM telemetry wrapper, mid-stream failover, and
  graceful degradation.
- **Data & Storage:** PostgreSQL (with Row-Level Security & `pgvector`), MongoDB (change streams for RAG sync), and
  Pydantic repositories.
- **Agent Core:** Explicit state-machine loop (hand-rolled and ported to LangGraph) with human-in-the-loop (HITL)
  approval gates and verbatim memory retrieval.
- **RAG Pipeline:** Hybrid search (`tsvector` + vector fused via RRF), re-ranking, query rewriting, and citation-backed
  groundedness checks.
- **Evaluations & Observability:** Deterministic DB-state golden evals, calibrated LLM-as-judge, OpenTelemetry tracing,
  and self-hosted Langfuse.
- **Security & Enterprise Integration:** Prompt-injection defenses, PII redaction, append-only audit logs, MCP server
  exposure, OAuth2 delegation, and Temporal durable workflows.

---

## Development Roadmap & Milestones

### Stage 0: Harden the Router

Establish a resilient, observable LLM gateway before introducing agentic complexity.

- [ ] **0.1 Unified LLM Wrapper:** Single call wrapper recording purpose, model, prompt/completion tokens, latency, and
  cost (including internal summarizer calls).
- [ ] **0.2 Compaction Race Fix:** Ensure context compaction removes only the exact messages that were summarized.
- [ ] **0.3 Service Architecture:** Replace the CLI loop with a FastAPI `/chat` SSE streaming endpoint.
- [ ] **0.4 Core Bug Fixes & Config:** Fix TTFT `None` crash, remove unreachable `exit(0)` after `raise`, and migrate
  configuration to typed settings.
- [ ] **0.5 Streaming Resilience:** Implement time-to-first-chunk (TTFC) and stall timeouts with mid-stream failover
  tested against a mock FastAPI provider.
- [ ] **0.6 Graceful Degradation:** Fallback path when all upstream models are unavailable (canned response + automated
  support ticket creation).

### Stage 1: Data Layer

Build a strictly isolated, realistic multi-tenant persistence layer.

- [ ] **1.1 Relational Schema:** Postgres schema and migrations for `tenants`, `customers`, `products`, `inventory`,
  `orders`, `order_items`, and `returns`.
- [ ] **1.2 Document Store:** MongoDB collections for `product_docs`, `reviews`, and `policies`.
- [ ] **1.3 Deterministic Seeding:** Seed ~500 products, ~200 customers, and ~1,000 orders with realistic data
  anomalies (null addresses, duplicate customers, stale inventory, conflicting policy versions).
- [ ] **1.4 Repository Layer:** Typed Pydantic repositories enforcing `tenant_id` on every query.
- [ ] **1.5 Defense-in-Depth Isolation:** Enforce PostgreSQL Row-Level Security (RLS) as a secondary tenant isolation
  boundary.

### Stage 2: Tools and Contracts

Define strict, resilient, least-privilege interfaces between the model and backend systems.

- [ ] **2.1 Read Tools:** Implement `get_order`, `search_products`, and `check_stock` with strict Pydantic input/output
  schemas.
- [ ] **2.2 Error Taxonomy:** Standardize structured errors returned to the model (`not_found`, `invalid_args`,
  `transient`, `forbidden`).
- [ ] **2.3 Idempotent Write Tools:** Implement `create_return` backed by idempotency keys.
- [ ] **2.4 Self-Healing Validation:** Argument validation layer that returns structured errors for malformed LLM
  arguments with a capped repair retry count.
- [ ] **2.5 Deterministic Tool Testing:** Comprehensive unit test suite verifying tool contracts without LLM calls.
- [ ] **2.6 Resilience Controls:** Per-tool timeouts, retries, and rate limits tested against a simulated flaky legacy
  backend.
- [ ] **2.7 Access Scopes:** Role-based tool permissions enforcing the principle of least privilege.

### Stage 3: Agent Core

Build a controllable, resumable reasoning loop with benchmarked memory strategies.

- [ ] **3.1 Bounded Execution Loop:** Hand-rolled `plan → tool → observe` loop constrained by max steps, token budget,
  and wall-clock timeout.
- [ ] **3.2 Explicit State Machine:** Formalized states, transitions, retries, and timeouts with an architecture state
  diagram in `/docs`.
- [ ] **3.3 Human-in-the-Loop (HITL):** Approval gate for refunds exceeding a configured threshold (
  `pause → persist → resume`).
- [ ] **3.4 Safe Tool Failover:** Discard partial tool-call chunks on mid-stream failure and re-run the step from
  scratch—never resume partial tool JSON.
- [ ] **3.5 Session Persistence:** Store conversation sessions in PostgreSQL with automated data retention policies.
- [ ] **3.6 Memory Benchmark:** Evaluate full history vs. rolling summary vs. verbatim store across total token
  consumption (including summarizer overhead) and recall score.
- [ ] **3.7 LangGraph Port & ADR:** Port the core loop to LangGraph and document an Architecture Decision Record (ADR)
  comparing abstractions vs. raw control.
- [ ] **3.8 Multi-Agent Evaluation:** Implement a supervisor/worker split *only* if empirical failure modes justify
  moving beyond a single agent.
- [ ] **3.9 Verbatim Memory Retrieval:** Implement targeted retrieval over a verbatim conversation store ("drawers"
  architecture).

### Stage 4: Retrieval-Augmented Generation (RAG)

Ground agent responses in versioned, tenant-scoped product and policy knowledge.

- [ ] **4.1 Vector Ingestion:** Chunk and embed policies, FAQs, and product documentation into `pgvector`.
- [ ] **4.2 Hybrid Search:** Combine `tsvector` keyword search and vector similarity using Reciprocal Rank Fusion (RRF).
- [ ] **4.3 Scoped Filtering:** Enforce metadata filtering by `tenant_id`, `locale`, and document `version`.
- [ ] **4.4 Retrieval Evaluation:** Benchmark two chunking strategies against a golden Q&A dataset using `Recall@k`,
  `Precision`, and `MRR`.
- [ ] **4.5 Retrieval Optimization:** Test query rewriting and cross-encoder re-ranking; retain only components that
  demonstrably improve eval metrics.
- [ ] **4.6 Real-Time Sync:** Keep embeddings fresh via MongoDB change streams.
- [ ] **4.7 Citation Grounding:** Enforce groundedness checks with inline source citations.

### Stage 5: Evaluations & Regression Testing

Shift from subjective "vibe checks" to deterministic state assertions and CI-gated quality metrics.

- [ ] **5.1 Golden Dataset:** ~50 end-to-end tasks specifying expected tool trajectories and expected final database
  states.
- [ ] **5.2 State-Based Grading:** Score task success via deterministic PostgreSQL state assertions rather than LLM
  judges.
- [ ] **5.3 CI Regression Gate:** Version prompts as files and fail CI builds if eval scores regress.
- [ ] **5.4 Calibrated LLM-as-Judge:** Evaluate subjective qualities (tone, groundedness) calibrated against ~20
  human-labeled examples with reported inter-annotator agreement.
- [ ] **5.5 Tier Shadow Testing:** Run Fast vs. Smart model tiers in shadow/A-B mode to compare task success, latency,
  and cost.
- [ ] **5.6 Adversarial Suite:** Build an adversarial prompt dataset feeding directly into Stage 7 security testing.

### Stage 6: Observability & Operations

Ensure full visibility into agent execution, cost, and failure modes.

- [ ] **6.1 Distributed Tracing:** Instrument OpenTelemetry traces (`run → step → LLM/tool spans`).
- [ ] **6.2 LLM Observability:** Integrate self-hosted Langfuse tracking token usage and cost per run.
- [ ] **6.3 Failure Taxonomy:** Categorize failures systematically across `model`, `tool`, `data`, and `policy`
  boundaries.
- [ ] **6.4 SLOs & Alerting:** Define Service Level Objectives (success rate, p95 latency, cost per task) backed by a
  dashboard and active alerting.
- [ ] **6.5 Chaos Engineering & Runbooks:** Induce a controlled failure, conduct a blameless postmortem, and author an
  operational runbook.

### Stage 7: Security & Governance

Harden the system against adversarial inputs, data leakage, and compliance violations.

- [ ] **7.1 Prompt Injection Defense:** Verify resistance against indirect prompt injection (e.g., malicious product
  reviews in MongoDB attempting unauthorized refunds).
- [ ] **7.2 Privacy & GDPR:** Implement automated PII redaction in logs/traces and a compliant GDPR customer-deletion
  workflow.
- [ ] **7.3 Guardrails:** Deploy input and output policy guardrails.
- [ ] **7.4 Tenant Isolation Testing:** Run automated cross-tenant exfiltration penetration tests.
- [ ] **7.5 Immutable Audit Trail:** Maintain an append-only audit log capturing *who*, *what*, *why*, and *approval
  reference* for state-mutating actions.
- [ ] **7.6 Explainable Decision Records:** Generate human-readable decision summaries per refund for non-engineering
  auditors and support leads.

### Stage 8: Enterprise Integration

Connect the agent ecosystem to standard enterprise protocols and durable workflows.

- [ ] **8.1 Model Context Protocol (MCP):** Expose agent tools via an MCP server utilizing `mcp-shield` and a Postgres
  MCP gateway.
- [ ] **8.2 Identity & Auth:** Support OAuth2 delegated user tokens flowing through to tool execution alongside service
  accounts for background jobs.
- [ ] **8.3 Durable Workflows:** Orchestrate long-running return lifecycles (approval waits, retries, compensating
  transactions) using Temporal.
- [ ] **8.4 ITSM/ERP Integration:** Integrate mock ticketing/ERP webhooks or queues for escalations with rate-limit
  handling and partial-failure recovery.

### Stage 9: Deploy & Operate

Package, validate under load, and deploy to production infrastructure.

- [ ] **9.1 Local Stack:** Unified `docker-compose` setup orchestrating the API, PostgreSQL, MongoDB, Temporal, and
  Langfuse.
- [ ] **9.2 CI Pipeline:** Automated pipeline executing linters, deterministic unit tests, and eval regressions.
- [ ] **9.3 Cloud Deployment:** Deploy to GCP (EU region) using Gemini models via Vertex AI.
- [ ] **9.4 Load Testing:** Benchmark p95 latency and Time-To-First-Token (TTFT) under realistic concurrent traffic.
- [ ] **9.5 Documentation & Handover:** Finalize Architecture Decision Records (ADRs) and operational handover
  documentation.

---

## Repository Structure (Planned)

```text
├── cli/                          # Interactive terminal chat client & CLI entrypoints
├── config.toml                   # Application, model, and runtime configuration
├── pyproject.toml                # Project metadata, build configuration, and dependencies
├── src/
│   └── shopassist/
│       ├── api/                  # FastAPI app, dependency injection, and SSE chat endpoints
│       │   └── routes/           # Modular route handlers (healthz, chat SSE, sessions, usage)
│       ├── agent/                # Hand-rolled agent loop, state machine, and LangGraph adapters
│       ├── core/                 # Shared domain models, typed errors, and base execution context
│       ├── data/                 # Repository layer, Postgres/pgvector RLS, and MongoDB models
│       ├── llm/                  # LLM gateway, LiteLLM router factory, failover, and token cost tracking
│       ├── mcp/                  # Model Context Protocol (MCP) server gateway & tool exposure
│       ├── memory/               # Conversation history, UUIDv7 messages, and async compactor
│       ├── rag/                  # Hybrid search (tsvector + vector), chunking, and re-rankers
│       ├── telemetry/            # OpenTelemetry distributed tracing, Langfuse hooks, and log redaction
│       ├── workflows/            # Temporal asynchronous workflows for long-running refund approvals
│       ├── main.py               # Application server entrypoint
│       ├── router_factory.py     # LiteLLM router assembly
│       ├── settings.py           # Typed environment and configuration loader
│       └── wiring.py             # Dependency injection container & build_gateway assembly
├── tests/                        # Unit, concurrency, security, and integration test suites
├── tools/                        # Local development utilities and fake LLM provider mock server
└── uv.lock                       # Deterministic dependency lockfile for uv