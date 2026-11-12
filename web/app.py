"""FastAPI web app for the Embodied AI Agent: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8132, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. Everything
else is local-only: the live agent (web/agent_control.py), the memory database browser, output logs and the Ollama
status (web/local_data.py). Only the live agent and the Ollama check talk to an LLM server, and only to the
api_base in config.yaml (localhost).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web import agent_control, introspect, local_data

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Embodied AI Agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


# ---- exported to the static site ------------------------------------------------
@app.get("/api/info")
def info():
    """Project summary and counts for the home page."""
    cfg = introspect.config_summary()
    mods = introspect.architecture()
    d = introspect.defaults()
    return {"title": "Embodied AI Agent", "description": introspect.readme_summary(),
            "modules": len(mods), "lines": sum(m["lines"] for m in mods),
            "classes": sum(len(m["classes"]) for m in mods),
            "platforms": cfg["platforms"], "default_platform": cfg["platform"],
            "sensors": {p: len(v) for p, v in cfg["sensors"].items()},
            "actuators": {p: len(v) for p, v in cfg["actuators"].items()},
            "models": len(cfg["models"]), "model_ids": sorted({m.get("model_id", "") for m in cfg["models"].values()}),
            "emotions": len(d["emotions"]), "thinking_modes": len(d["thinking_modes"]),
            "tables": len(introspect.memory_schema())}


@app.get("/api/config")
def config():
    """config.yaml (platforms, sensors, actuators, models, intervals), as loaded by utils.config.load_config."""
    return introspect.config_summary()


@app.get("/api/config_yaml")
def config_yaml():
    return {"text": introspect.raw_config_text()}


@app.get("/api/defaults")
def defaults():
    """Initial emotions, drives, topics, thinking modes and other constants, read from the real modules."""
    return introspect.defaults()


@app.get("/api/architecture")
def architecture():
    """Every module with its classes, functions and internal imports (parsed with ast)."""
    return introspect.architecture()


@app.get("/api/schema")
def schema():
    """SQLite tables created by MemoryManager."""
    return introspect.memory_schema()


# ---- local only -------------------------------------------------------------------
@app.get("/api/local/ollama")
def ollama():
    """Is the local Ollama server up, and which configured models does it have?"""
    return local_data.ollama_status(introspect.config_summary()["models"])


@app.get("/api/local/memory")
def memory_stats():
    return local_data.memory_overview()


@app.get("/api/local/memory/{table}")
def memory_table(table: str, limit: int = 50):
    try:
        return local_data.memory_rows(table, limit)
    except KeyError:
        raise HTTPException(404, f"unknown table {table}")


@app.get("/api/local/logs")
def logs(lines: int = 60):
    return local_data.output_logs(lines)


class StartBody(BaseModel):
    platform: str = "mac"


class MessageBody(BaseModel):
    text: str


class ActionBody(BaseModel):
    type: str
    target: str = "all"
    parameters: dict = {}
    description: str = ""


@app.get("/api/local/agent")
def agent_state():
    return agent_control.state()


@app.post("/api/local/agent/start")
def agent_start(body: StartBody):
    try:
        return agent_control.start(body.platform)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/local/agent/stop")
def agent_stop():
    return agent_control.stop()


@app.post("/api/local/agent/message")
def agent_message(body: MessageBody):
    try:
        return agent_control.message(body.text)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.post("/api/local/agent/action")
def agent_action(body: ActionBody):
    try:
        return agent_control.action(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/config", "/config_yaml", "/defaults", "/architecture", "/schema"]
