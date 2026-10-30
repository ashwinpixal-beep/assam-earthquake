#!/usr/bin/env python3
"""One-click launcher for the damage-risk site.

Installs the Python requirements if they are missing, starts the FastAPI app on
http://127.0.0.1:8000 and opens the browser. Ctrl+C stops the server.

    python3 start.py                 # default port 8000
    python3 start.py --port 8010     # different port
    python3 start.py --no-browser    # headless / server use
    python3 start.py --host 0.0.0.0  # reachable from other machines
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import subprocess
import sys
import time
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["fastapi", "uvicorn", "joblib", "numpy", "pandas", "sklearn", "lightgbm", "xgboost"]
IMPORT_NAMES = {"sklearn": "scikit-learn"}


def missing_packages() -> list[str]:
    missing = []
    for mod in REQUIRED:
        if importlib.util.find_spec(mod) is None:
            missing.append(IMPORT_NAMES.get(mod, mod))
    return missing


def install(packages: list[str]) -> None:
    print("Installing:", ", ".join(packages))
    attempts = [
        [sys.executable, "-m", "pip", "install", "--quiet", "-r",
         os.path.join(HERE, "requirements.txt")],
        [sys.executable, "-m", "pip", "install", "--quiet", "--user",
         "-r", os.path.join(HERE, "requirements.txt")],
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--break-system-packages", "-r", os.path.join(HERE, "requirements.txt")],
    ]
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, cwd=HERE)
            return
        except subprocess.CalledProcessError:
            continue
    sys.exit(
        "Could not install the requirements automatically.\n"
        "Create a virtual environment and try again:\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        "  .venv/bin/python start.py"
    )


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host if host != "0.0.0.0" else "", port))
            return True
        except OSError:
            return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Start the damage-risk site.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(HERE, "models", "damage_model.joblib")):
        print("Warning: models/damage_model.joblib is missing, so /api/predict will return 503.")
    if not os.path.exists(os.path.join(HERE, "public", "static", "hazard", "hazard_grid.geojson")):
        print("Warning: hazard data is missing. Run: python3 scripts/build_hazard.py")

    missing = missing_packages()
    if missing and not args.skip_install:
        install(missing)
        missing = missing_packages()
        if missing:
            print("Still missing after install:", ", ".join(missing))

    port = args.port
    while not port_is_free(args.host, port) and port < args.port + 20:
        print(f"Port {port} is busy, trying {port + 1}. (Use --port to choose one.)")
        port += 1

    url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", args.host, "--port", str(port),
    ]
    print(f"\n  damage-risk site\n  pages: {url}/  ·  {url}/2d  ·  {url}/docs\n  Ctrl+C to stop\n")

    env = dict(os.environ, PYTHONPATH=HERE)
    proc = subprocess.Popen(cmd, cwd=HERE, env=env)
    try:
        if not args.no_browser:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()
