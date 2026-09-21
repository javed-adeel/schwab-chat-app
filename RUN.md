# How to Run

**Requires:** Python 3.10+

## 1. Install (one time)

Open a terminal in the project folder (the one containing `pyproject.toml`).

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

> PowerShell blocks activation? Run `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again.

## 2. Set your API key

Use any one provider:

| Provider | Key variable | Default model |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-3.6-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-5` |

Set it one of two ways:

- **Current terminal only**
  - Windows: `$env:GEMINI_API_KEY = "..."`
  - macOS/Linux: `export GEMINI_API_KEY=...`
- **Saved in a file:** copy `.env.example` to `.env` and fill in the matching line. Never commit `.env`.

If more than one key is set, the app uses the first in the order Anthropic, Gemini, OpenRouter.
To choose explicitly, set `NEWSCHAT_LLM_PROVIDER=gemini` (or `anthropic`, `openrouter`, `extractive`).

No key? The app still runs in extractive mode (quotes sentences, no summaries).

## 3. Start the app

```
python -m newschat
```

Open http://127.0.0.1:8000

The status line shows the provider and model in use, e.g. `118 articles indexed · gemini:gemini-3.6-flash`.
If it says `extractive mode`, no key was picked up.

## 4. Run the checks (optional)

```
python -m pytest                       # 254 tests + coverage gate
python -m newschat.evaluation.cli      # retrieval quality report
python -m newschat.evaluation.cli --ablation
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named newschat` | Run step 1 again, inside the activated `.venv`. |
| `dataset not found` | Start the app from the folder containing `pyproject.toml`. |
| "Language model temporarily unavailable" | The real error is printed in the terminal running the server (bad key, wrong model, rate limit). |
| Status line shows a different provider than expected | With several keys set, `auto` picks Anthropic, then Gemini, then OpenRouter. Set `NEWSCHAT_LLM_PROVIDER` to choose. |
| Startup fails with `... but no API key is set` | `NEWSCHAT_LLM_PROVIDER` names a provider whose key is missing. Add the key or change the provider. |
| `install the 'gemini' extra ...` (or `openrouter` / `anthropic`) | That provider's SDK isn't installed. Run `python -m pip install -e ".[dev]"` (all providers) or the named extra. |
| Gemini answers stop mid-sentence | Thinking models spend part of the token budget reasoning. Raise `NEWSCHAT_LLM_MAX_TOKENS` (e.g. `2000`). |
| Port 8000 in use | Set `NEWSCHAT_PORT=8001` before starting. |

## Optional settings

| Variable | Purpose | Default |
|---|---|---|
| `NEWSCHAT_LLM_PROVIDER` | `auto`, `anthropic`, `gemini`, `openrouter`, `extractive` | `auto` |
| `NEWSCHAT_ANTHROPIC_MODEL` | Anthropic model | `claude-sonnet-5` |
| `NEWSCHAT_GEMINI_MODEL` | Gemini model | `gemini-3.6-flash` |
| `NEWSCHAT_OPENROUTER_MODEL` | OpenRouter model slug | `anthropic/claude-sonnet-5` |
| `NEWSCHAT_LLM_MAX_TOKENS` | Answer length cap | `900` |
| `NEWSCHAT_OPENROUTER_BASE_URL` | OpenRouter endpoint (any OpenAI-compatible URL works) | `https://openrouter.ai/api/v1` |
| `NEWSCHAT_PORT` | Server port | `8000` |
| `NEWSCHAT_LOG_LEVEL` | Log detail | `INFO` |
