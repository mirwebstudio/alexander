# Alexander

Alexander is a passphrase-protected personal assistant with a Streamlit web
interface and an optional FastAPI dashboard. The core also supports a
dependency-light demo mode, an Ollama-backed model, local notes/documents,
memory, optional voice tools, and CLI evaluations.

## Requirements and installation

Python 3.10 or later is recommended.

```bash
cd ALEXANDER_AI_FINAL
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Set a private passphrase of at least 12 characters in the environment. Do not
put it in source control:

```bash
export ALEXANDER_PASSWORD='choose-a-long-private-passphrase'
```

PowerShell:

```powershell
$env:ALEXANDER_PASSWORD = "choose-a-long-private-passphrase"
```

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `ALEXANDER_PASSWORD` | Required owner passphrase; at least 12 characters. Configure as a hosting secret. | unset |
| `ALEXANDER_MODEL` | Ollama chat model | `llama3.1:8b` |
| `ALEXANDER_EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `ALEXANDER_OWNER_NAME` | Name used when addressing the owner | `sir` |
| `ALEXANDER_HOST` | Host for the optional FastAPI server | `127.0.0.1` |
| `ALEXANDER_PORT` | Port for the optional FastAPI server | `8000` (or platform `PORT`) |
| `ALEXANDER_COOKIE_SECURE` | Set to `false` only for local HTTP use of FastAPI | `true` |
| `ALEXANDER_ALLOWED_HOSTS` | Comma-separated additional FastAPI hostnames | loopback hosts only |

`OLLAMA_HOST` may be set using the Ollama client's configuration when Ollama
runs on a different machine. Do not put credentials in source files or commit
local secrets.

## Run the Streamlit app

```bash
python -m streamlit run app.py
```

The app prompts for the passphrase before creating a user session. Each
Streamlit browser session gets its own conversation, assistant, and memory
file. Use **Lock** to return to the passphrase screen.

### GitHub Codespaces

The repository root has a small `app.py` launcher so the command below works
from the default Codespaces workspace directory. It delegates to the single
application implementation in this folder. The root Streamlit config binds
to `0.0.0.0:8501`, and `.devcontainer/devcontainer.json` forwards port 8501
and opens the browser when the port becomes available.

```bash
python -m streamlit run app.py
```

The first Codespace created with this configuration installs the project
requirements automatically. If this is an existing Codespace, rebuild it
once so the `forwardPorts` setting takes effect, or install dependencies with
`python -m pip install -r ALEXANDER_AI_FINAL/requirements.txt`.

Configure `ALEXANDER_PASSWORD` as a Codespaces secret, then restart the
Codespace after adding or changing that secret. The forwarded site displays
the passphrase login form; if the secret is missing, the app displays a
configuration error instead of unlocking.

## Optional local model and features

Without the optional `ollama` Python package, Alexander uses its built-in
demo responses. For local model-backed answers, install Ollama, pull a model,
then install the Python client:

```bash
ollama pull llama3.1:8b
python -m pip install ollama
```

The machine running Alexander must be able to reach the Ollama service.
Setting `OLLAMA_HOST` is supported by the Ollama client when Ollama is hosted
elsewhere. Do not assume a cloud host's `localhost` is your personal computer.

Other optional packages:

```bash
python -m pip install chromadb pypdf
python -m pip install numpy sounddevice webrtcvad faster-whisper soundfile
python -m pip install torch
```

These enable vector memory/knowledge indexing, PDF extraction, voice mode,
and TinyGPT training, respectively. Voice additionally requires a working
audio device; Piper can be installed separately for offline speech output.

Put personal notes in `notes/` and documents in `docs/`. Do not add private
documents to a public Git repository.

## Optional FastAPI dashboard

The FastAPI dashboard uses the same passphrase and has HTTP-only, strict
session cookies, session expiry, request validation, and no permissive CORS.
It binds to loopback by default:

```bash
export ALEXANDER_COOKIE_SECURE=false  # local HTTP only
python alexander.py --serve
```

Open <http://127.0.0.1:8000>. For HTTPS deployments, leave
`ALEXANDER_COOKIE_SECURE` unset (secure cookies are enabled by default).
Changing `ALEXANDER_HOST` from `127.0.0.1` exposes the service to other
interfaces; use HTTPS, set `ALEXANDER_ALLOWED_HOSTS` to the required host
names, and keep the passphrase private. The API does not expose its OpenAPI
schema or interactive docs.

## Deploying on Render

The repository-root `render.yaml` points Render at `ALEXANDER_AI_FINAL` and
deploys the Streamlit interface. Set the prompted `ALEXANDER_PASSWORD`
secret in Render before making the service available.
The deployment uses a shared owner passphrase: anyone who knows it can use
the app. Notes and documents in `notes/` and `docs/` are shared with every
authenticated visitor; only per-session conversation memory is isolated.
Never deploy confidential notes or documents to an instance accessible to
other people.

Render's local filesystem may be ephemeral. Back up any data you need to keep
and do not treat a free hosted instance as durable storage. The app does not
encrypt files at rest.

## Privacy and network behavior

- Passphrases are read from `ALEXANDER_PASSWORD`; they are not stored in code
  or logged.
- Model inference uses the configured Ollama service when available. The
  demo fallback is not a language model.
- Web search can send the search query to SearXNG at `localhost:8080` or to
  DuckDuckGo. Do not include private data in a search query.
- Uploaded/shared notes and documents remain on the host's local filesystem;
  hosting providers can still access that filesystem.
- The FastAPI dashboard is loopback-only unless you explicitly change
  `ALEXANDER_HOST`.

## Checks

Run the built-in self-test:

```bash
python alexander.py --selftest
```

Run the regression suite:

```bash
python -m unittest discover -s tests -v
```
