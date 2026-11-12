"""Web dashboard and static-site source for the Embodied AI Agent.

Modules:
    app -- FastAPI app: /api routes, STATIC_ENDPOINTS exported to the GitHub Pages site
    introspect -- facts read from the code and config (modules via ast, SQLite schema, cognitive defaults)
    local_data -- local-only: memory.db browser, output logs, Ollama status
    agent_control -- local-only: run the real EmbodiedAgent (simulated sensors) from the browser
    static/ -- ES-module front end; *_engine.js are JS ports of the non-LLM logic (node-tested)
"""
