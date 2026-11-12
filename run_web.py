"""Start the Embodied AI Agent web dashboard on http://127.0.0.1:8132 (local only). Requires requirements-web.txt."""
import os
import sys

import uvicorn

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    sys.path.insert(0, root)
    uvicorn.run("web.app:app", host="127.0.0.1", port=8132)
