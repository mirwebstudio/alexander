# Alexander AI

The application source, setup instructions, and tests are in
[`ALEXANDER_AI_FINAL/`](ALEXANDER_AI_FINAL/). The Render deployment manifest
is kept at the repository root so Render can discover it; it sets
`ALEXANDER_AI_FINAL` as the service root directory.

See [the project README](ALEXANDER_AI_FINAL/README.md) for local setup and
deployment instructions. Configure `ALEXANDER_PASSWORD` as a secret in the
hosting provider before deploying. The root `app.py` delegates to the single
implementation in the project folder, and `.devcontainer/devcontainer.json`
configures Codespaces to forward port 8501 and open the forwarded site.
