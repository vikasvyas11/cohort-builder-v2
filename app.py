"""Start the API and web UI with `python app.py`.

On a laptop, in Docker or on a VM this is plain uvicorn, the same as `uvicorn api.main:app`.

On a Hugging Face Space (the Gradio SDK) it is different, and for a reason worth knowing. Hugging Face decides a
Space has started by watching Gradio. On ZeroGPU hardware it is stricter: the `spaces` package only reports a
successful start from inside `gradio.Blocks.launch()`, and stops any Space that has not made that call. So on a
Space this app lets Gradio own the server on port 7860, exactly as any Gradio app does, and hands Gradio this
app's routes, error handlers and CORS settings, so the web UI and the API are served from that same server.
The Gradio interface itself is empty; nobody sees it. If anything about that fails, it says exactly what and
falls back to uvicorn.

The `spaces` import comes first on purpose: the package must load before anything that touches CUDA.
"""

import traceback

try:
    import spaces
except ImportError as _spaces_error:
    spaces, SPACES_ERROR = None, _spaces_error
else:
    SPACES_ERROR = None

import importlib.util
import inspect
import os
import sys

import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from api.main import app, origins, revalidate_ui_files

if spaces is not None:
    @spaces.GPU
    def _gpu_placeholder() -> None:
        """Never called. ZeroGPU refuses to start a Space that defines no GPU function."""
        return None


def on_hugging_face_space() -> bool:
    return bool(os.environ.get("SPACE_ID"))


def describe_environment() -> None:
    """One short block in the log that says what this container is, so a failed start can be diagnosed."""
    def version(name: str) -> str:
        try:
            import importlib.metadata as metadata
            return metadata.version(name)
        except Exception:                                       # noqa: BLE001 - only for the log
            return "not installed"

    print("Environment: python " + sys.version.split()[0]
          + ", gradio " + version("gradio") + ", starlette " + version("starlette") + ", fastapi " + version("fastapi")
          + ", spaces " + (version("spaces") if spaces is not None else f"unavailable ({SPACES_ERROR})")
          + ", torch " + ("present" if importlib.util.find_spec("torch") else "absent")
          + f", ZeroGPU={os.environ.get('SPACES_ZERO_GPU', 'no')}, space={os.environ.get('SPACE_ID', 'no')}", flush=True)


def _gradio_app_kwargs() -> dict:
    """Make Gradio's own FastAPI app carry this app's routes, error handlers and CORS."""
    ours = [r for r in app.router.routes if getattr(r, "path", "").startswith(("/api", "/js"))
            or getattr(r, "path", "") in ("/", "/style.css", "/config.js")]
    return {"routes": ours, "exception_handlers": dict(app.exception_handlers),
            "middleware": [Middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]),
                           Middleware(BaseHTTPMiddleware, dispatch=revalidate_ui_files)]}


def serve_with_gradio(port: int) -> None:
    import gradio as gr
    with gr.Blocks(title="Cohort Builder") as demo:
        gr.Markdown("Cohort Builder is served at this Space's main address.")
        if spaces is not None:
            hidden = gr.Button(visible=False)
            hidden.click(_gpu_placeholder, None, None)
    options = {"server_name": "0.0.0.0", "server_port": port, "app_kwargs": _gradio_app_kwargs()}
    if "ssr_mode" in inspect.signature(demo.launch).parameters:
        options["ssr_mode"] = False          # no Node server in front: the UI is plain static files
    demo.launch(**options)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    # One worker on purpose: sessions live in this process's memory.
    if on_hugging_face_space() or os.environ.get("COHORT_BUILDER_USE_GRADIO"):
        describe_environment()
        if spaces is None and os.environ.get("SPACES_ZERO_GPU"):
            print("WARNING: this is a ZeroGPU Space but the `spaces` package failed to import, so no GPU function can be "
                  "registered and Hugging Face will stop the Space. The reason: " + str(SPACES_ERROR), flush=True)
        try:
            serve_with_gradio(port)
        except Exception:                                       # noqa: BLE001 - the app should still come up
            print("Gradio could not start the app. The full error follows; serving with uvicorn instead.", flush=True)
            traceback.print_exc()
        else:
            raise SystemExit(0)
    uvicorn.run(app, host="0.0.0.0", port=port)
