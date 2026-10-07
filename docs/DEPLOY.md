# Deploying for free

The app is two parts that can live in different places:

| Part | What it is | Free home |
|---|---|---|
| **API** (`api/`, `modules/`) | Python: FastAPI, Splink, DuckDB | A free ZeroGPU Gradio Space on Hugging Face, or a free VM |
| **Web UI** (`web/`) | Static files, no build step | The same host (the API serves it), or Cloudflare Pages |

Cloudflare's free plan cannot run the Python engine (Workers allow a few milliseconds of CPU and about 128 MB of
memory), which is why the engine runs elsewhere and Cloudflare only serves the page.

Free-tier terms change. Everything about Hugging Face below comes from its own documentation (pages "Spaces ZeroGPU",
"Using GPU Spaces" and "Spaces Configuration Reference"); the other providers' terms I could not check.

## 1. A free Hugging Face Space (the ZeroGPU route)

### What Hugging Face's documentation says, and what follows from it

- "CPU Basic has no hourly cost, but creating a new Space that runs on compute (Gradio or Docker) requires a paid plan.
  Static Spaces are free for everyone." So a free account cannot pick CPU basic for a new Gradio or Docker Space. That
  is expected, not a fault: stop trying to.
- Free personal accounts "in good standing (verified email, account older than 30 days) can host up to 2 ZeroGPU
  Spaces for free". That is the free compute on offer, and it is what your new Space is on.
- ZeroGPU Spaces are "exclusively compatible with the Gradio SDK" (Gradio 4 or later), with Python 3.12.12 or 3.10.13.
- A ZeroGPU Space is only started if the code defines a `@spaces.GPU` function and the app calls Gradio's `launch()`.
  `@spaces.GPU` "is designed to be effect-free in non-ZeroGPU environments". This app never uses a GPU, so the
  function is an unused placeholder; GPU quota is never spent.

The app is built for exactly that, in `app.py`:

- On a Space (the `SPACE_ID` variable is set) Gradio owns the server on port 7860 and is handed this app's routes, error
  handlers and CORS settings. The web UI and the API are served from that one server. The Gradio page itself is empty.
- `spaces` is imported first, and the unused GPU function exists.
- The log begins with an `Environment:` line (Python, gradio, starlette, fastapi, spaces and torch versions, ZeroGPU flag)
  and any failure is printed in full instead of being summarised. If Gradio fails to start the app falls back to
  uvicorn and says why.

### The one thing that broke earlier attempts

The Space installs Gradio at the README's `sdk_version`, then installs `requirements.txt`. The first instructions here
said `sdk_version: 5.0.0`, and Gradio 5.0.0 no longer imports against today's libraries: in a clean environment,
`import gradio` fails with `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`. An older version of
`app.py` caught every `ImportError` and printed "Gradio is not installed", which hid this. Gradio was installed; it was
broken, so the Space fell back to plain uvicorn, never called `launch()`, and ZeroGPU stopped it.
The cure is the pattern the Cohort Builder Assistant Space already uses: a current Gradio, the same version in
`sdk_version` **and** in `requirements.txt` (6.20.0). Updating `app.py` alone is not enough; the README and requirements
must change too, which is why the bundle replaces all three.

### Steps

1. In the project, run `python tools/make_space_bundle.py`. It writes `cohort-builder-space/` next to the project:
   `app.py`, `api/`, `modules/`, `utils/`, `web/`, a `requirements.txt` that includes `gradio==6.20.0` and
   `spaces>=0.51.0`, and a `README.md` with the Space header (`sdk: gradio`, `sdk_version: 6.20.0`,
   `python_version: "3.12.12"`, `app_file: app.py`).
2. Copy **everything** in that folder to the top level of your Space's git repo, replacing the files already there,
   `README.md` and `requirements.txt` included. Delete any `Dockerfile` or leftover files from earlier attempts.
3. Commit and push. Open the Space's **Logs** and look for `Environment:` followed by `Running on local URL`, with no
   `Shutting down` afterwards. Then open `https://<your-user>-<space-name>.hf.space`.
4. Optional settings under **Settings, Variables**: `COHORT_BUILDER_MAX_SESSIONS`,
   `COHORT_BUILDER_DUCKDB_MEMORY_LIMIT`, `COHORT_BUILDER_MAX_CANDIDATE_PAIRS` (see the README table).
   The pair limit defaults to 5,000,000 and a 50,000-record flow reaches it quickly if a loose rule such as a lone
   `first_name` is on. It measured about 0.6 KB per candidate pair (4.4 million pairs peaked at 2.7 GB), so a Space with
   16 GB can take 10,000,000: set `COHORT_BUILDER_MAX_CANDIDATE_PAIRS=10000000`, run the biggest flow you expect, and
   watch whether the Space restarts. The Configure step now names the biggest rules when a configuration is near the
   limit, which is usually the better fix than raising it.

Keep one worker: sessions live in the process's memory.

### If it still does not start

Send the log from `Application Startup` onward; the `Environment:` line and any `Gradio could not start the app` traceback
say what is wrong. Likely causes and cures:

- `ResolutionImpossible` while building: `requirements.txt` and `sdk_version` name different Gradio versions. Use the
  bundle's files unchanged.
- `Gradio could not start the app` followed by a traceback: the traceback is the answer. Send it.
- `WARNING: this is a ZeroGPU Space but the spaces package failed to import`: the reason follows on the same line.
  Adding `torch` to `requirements.txt` is the usual fix, since ZeroGPU supports torch 2.8 and later.
- The Space starts but the page is blank: open the browser console; a message about a script or `config.js` is the clue.
- The ZeroGPU allowance (2 Spaces, account older than 30 days with a verified email) is used up: delete an old Space.

## 2. A free VM you control (Oracle Cloud Always Free)

If the Space route is not enough, Oracle's Always Free tier has offered an ARM VM with up to 4 cores and 24 GB of RAM,
and it does not sleep. It needs a card for identity checks and some server upkeep.

1. Create the VM (Ubuntu), then `sudo apt install python3.12 python3.12-venv git` (or the nearest Python 3.12 build).
2. Clone the repo, `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`.
3. Run it as a service: `.venv/bin/python app.py` (set `PORT` to change the port).
4. Publish it without opening ports with a free Cloudflare Tunnel (`cloudflared tunnel`), pointing a name such as
   `app.yourdomain.com` at `http://localhost:7860`. This also puts the app under your own domain.

## 3. Your own computer

`python app.py` (or `uvicorn api.main:app --port 7860`) runs the whole app locally. A free Cloudflare Tunnel can show it
to others while your computer is on.

## 4. Docker hosts

The Dockerfile works anywhere that runs containers (Google Cloud Run, Fly.io, Render). These generally need a card or
have small free memory: Render's free tier is 512 MB, too small for 50,000 records. Hugging Face Docker Spaces need a
paid plan to create.

## 5. The web UI on your Cloudflare site (optional)

Serving the UI from the API's host is enough. To put it under your own domain instead:

1. In `web/config.js` set the API address:

   ```js
   window.COHORT_API = "https://<your-user>-<space-name>.hf.space";
   ```

2. In Cloudflare, **Workers & Pages, Create, Pages, Upload assets** (or connect the Git repo) and publish the `web/`
   folder. No build command; the output directory is `web`.
3. Add a custom domain such as `cohort.yourdomain.com` to the Pages project.
4. Tell the API which site may call it: set the variable `COHORT_BUILDER_CORS_ORIGINS=https://cohort.yourdomain.com`
   (comma-separate several). The default `*` also works; restricting it is tidier.

## Check it at scale

On the host run `python tools/scale_check.py 50000 6` from a shell with the project and requirements installed. It drives
the API in-process at 50,000 records and prints the time and memory of every step; the second argument is the memory cap
in GB at which it stops itself. Size the memory to the data: a loose blocking rule can use far more than the 50,000-record
default flow (under 1 GB), which is what `COHORT_BUILDER_MAX_CANDIDATE_PAIRS` is for.

## After deploying

Add the live address to the top of the README so visitors can open the app from GitHub.
