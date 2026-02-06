# Discord Local AI (v1)

A fully local Discord AI system with persistent memory, retrieval, reasoning, and reflection. Designed to run offline except for Discord connectivity.

## Project Tree

```
.
├── README.md
├── config.yaml
├── main.py
├── discord_gateway.py
├── model_interface.py
├── persona.py
├── memory_service.py
├── retrieval_engine.py
├── reasoning_engine.py
├── belief_tracker.py
├── reflection_worker.py
├── scheduler.py
├── scripts
│   └── migrate.py
└── requirements.txt
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Update `config.yaml` with your Discord token and admin user IDs.
3. Run migrations (creates the SQLite schema):
   ```bash
   python scripts/migrate.py --sqlite ./data/memory.db
   ```
4. Start the bot:
   ```bash
   python main.py
   ```

## Database Schema

SQLite table `memories` stores all memory types (episodic, semantic, summary, belief, curiosity):

- `id` (TEXT, primary key)
- `content` (TEXT)
- `type` (TEXT)
- `confidence` (REAL)
- `importance` (REAL)
- `timestamp` (TEXT, ISO8601)
- `user_id` (TEXT)
- `source_memory_ids` (TEXT, JSON array)

Vector storage is handled by Chroma in local persistent mode (`storage.chroma_path`).

## Local Model Server Integration

The model interface is abstracted in `model_interface.py`. The default provider is `local_http`, which sends OpenAI-compatible requests to a local endpoint (e.g., Ollama, LM Studio, llama.cpp server). Configure:

```yaml
model:
  provider: "local_http"
  endpoint_url: "http://localhost:8000/v1/chat/completions"
  model_name: "local-model"
```

If the endpoint is unavailable, a local fallback response is used to keep the bot functional offline.

## Admin / Safety Commands

- `!memory [n]` (admin) - list recent memories
- `!memory_delete <id>` (admin) - delete a memory
- `!wipe_user <id>` (admin) - remove all memory entries for a user
- `!memory_help` - show commands

## Notes

- Reflection runs locally on a timer to derive summaries and beliefs.
- The scheduler controls proactive messages and uses inactivity and curiosity levels.
- All memory is stored locally in SQLite and Chroma.
- Transparency logs are written to the SQLite `transparency_logs` table and to `logging.transparency_log_path`.
