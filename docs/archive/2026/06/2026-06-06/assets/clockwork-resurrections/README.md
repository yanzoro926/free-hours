# Clockwork Resurrections / 钟表复活

> *"Workflows that survive the crash of the machine."*

**Durable Mini** is a minimal durable execution engine in Python — workflows persist their state to SQLite and resume from where they left off after any crash or failure.

Inspired by [Microsoft's pg_durable](https://github.com/microsoft/pg_durable) (open-sourced June 2026).

## The Core Idea

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Step 1  │───▶│ Step 2  │───▶│ Step 3  │
│  ✅     │    │  ✅     │    │  💥     │
└─────────┘    └─────────┘    └─────────┘
                                  │
                    ┌─────────────┘
                    ▼
              ┌─────────┐    ┌─────────┐
              │ Step 1  │    │ Step 3  │
              │ (SKIP)  │───▶│  ✅     │
              └─────────┘    └─────────┘
              ┌─────────┐
              │ Step 2  │
              │ (SKIP)  │
              └─────────┘

After a crash, completed steps are NEVER re-executed.
The database is the source of truth.
```

## Installation

```bash
pip install -e .
# No dependencies beyond Python 3.10+ standard library
```

## Quick Start

```python
from durable_mini import DurableEngine, Workflow

engine = DurableEngine("my_workflows.db")

@engine.step()
def fetch_data():
    return [1, 2, 3, 4, 5]

@engine.step()
def process(item: int):
    return item * 2

@engine.step()
def aggregate(results):
    return {"sum": sum(results), "count": len(results)}

# Build the workflow DAG
wf = Workflow("my_pipeline")
wf.add_step(fetch_data)
wf.add_step(process, depends_on=["fetch_data"], fan_out=True)
wf.add_step(aggregate, depends_on=["process"])

# Run — survives crashes!
result = engine.run(wf)
print(result.output)  # {'sum': 30, 'count': 5}
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Durable Execution** | Every step's inputs/outputs are checkpointed to SQLite |
| **Crash Recovery** | Resume from the last completed step — no re-execution |
| **Fan-out Parallelism** | Steps can fan out over list inputs (map pattern) |
| **Workflow DAG** | Define workflows as directed acyclic graphs |
| **Idempotent Replay** | Completed steps are immutable and never re-run |
| **Zero Dependencies** | Pure Python stdlib + SQLite |
| **Visualization** | ASCII DAG + timeline + HTML dashboard |

## Examples

```bash
# Basic data pipeline
python examples/basic_pipeline.py

# AI embedding pipeline (ingest → embed → index → search)
python examples/ai_pipeline.py

# Crash recovery demo
python examples/crash_recovery.py
```

## Visualization

```bash
# ASCII dashboard
python visualize.py my_workflows.db <instance_id> --format dashboard

# HTML report
python visualize.py my_workflows.db <instance_id> --format html -o report.html
```

## Architecture

```
┌──────────────────────────────────────┐
│           DurableEngine              │
│  ┌────────┐  ┌────────┐  ┌───────┐  │
│  │  Run   │  │ Resume │  │Status │  │
│  └───┬────┘  └───┬────┘  └───┬───┘  │
│      │           │            │       │
│  ┌───▼───────────▼────────────▼───┐  │
│  │        DurableBackend          │  │
│  │  ┌─────────────────────────┐   │  │
│  │  │        SQLite            │   │  │
│  │  │  • workflows             │   │  │
│  │  │  • instances             │   │  │
│  │  │  • steps (checkpoints)   │   │  │
│  │  └─────────────────────────┘   │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## How It Works

1. **Define**: Build a workflow DAG with `Workflow.add_step()`
2. **Execute**: `engine.run()` executes each step in topological order
3. **Checkpoint**: Before and after each step, state is written to SQLite
4. **Crash**: If the process dies, the database remains
5. **Resume**: Call `engine.run()` with the same `instance_id` — completed steps are skipped

The key insight: **each step is idempotent from the engine's perspective** because inputs and outputs are deterministically checkpointed.

## Compared to...

| System | Durable Mini | Temporal | pg_durable | Airflow |
|--------|:-----------:|:--------:|:----------:|:-------:|
| Language | Python | Go/Java/TS | SQL | Python |
| Database | SQLite | Cassandra/MySQL | PostgreSQL | PostgreSQL |
| Scale | Single process | Distributed | Database-native | Distributed |
| Setup | Zero | Complex | PG extension | Complex |
| Learning curve | Minimal | Steep | SQL | Moderate |

Durable Mini is a **learning tool and lightweight engine** — not a production orchestrator. Use Temporal or pg_durable for production workloads.

## Project Structure

```
clockwork-resurrections/
├── durable_mini/          # Core library
│   ├── __init__.py
│   ├── engine.py          # Durable execution engine
│   ├── workflow.py        # Workflow DAG definition
│   ├── backend.py         # SQLite persistence layer
│   └── decorators.py      # @step decorator
├── examples/
│   ├── basic_pipeline.py  # Data pipeline with crash recovery
│   ├── ai_pipeline.py     # AI embedding pipeline
│   └── crash_recovery.py  # Crash recovery demo
├── visualize.py           # Visualization tool
├── workflow_viz.html      # Sample HTML visualization
└── README.md
```

## License

MIT — built for exploration and learning.
