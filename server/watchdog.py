"""
Keeps the FastAPI server running continuously.

Launches `uvicorn main:app` (no --reload) as a subprocess, periodically
checks /health, and restarts the server if the process exits or stops
responding. Intended for keeping the app up outside of a proper process
manager (systemd, a host's own restart policy, etc.) — use one of those
instead if available; this is the "just keep it running" fallback.

Run from the server/ directory:
    python watchdog.py
"""
import subprocess
import sys
import time
import urllib.error
import urllib.request

HEALTH_URL = "http://127.0.0.1:8000/health"
CHECK_INTERVAL_SECONDS = 15
STARTUP_GRACE_SECONDS = 40
RESTART_DELAY_SECONDS = 3


def is_healthy():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run_forever():
    while True:
        print("[watchdog] starting server (uvicorn, no --reload)...", flush=True)
        # Deliberately bypass main.py's own --reload path: this watchdog *is*
        # the restart mechanism. Running uvicorn's file-watcher underneath a
        # second, external supervisor means two things are trying to manage
        # the same process lifecycle — when this script kills the reload
        # supervisor mid-restart, its worker (a separate process linked by an
        # OS pipe) crashes instead of exiting cleanly. A single plain worker
        # process, managed only by this script, avoids that entirely.
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
        started_at = time.time()

        while True:
            time.sleep(CHECK_INTERVAL_SECONDS)

            if proc.poll() is not None:
                print(f"[watchdog] server process exited (code {proc.returncode}) — restarting", flush=True)
                break

            past_grace_period = (time.time() - started_at) > STARTUP_GRACE_SECONDS
            if past_grace_period and not is_healthy():
                print("[watchdog] health check failed — restarting server", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break

        time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    try:
        run_forever()
    except KeyboardInterrupt:
        print("\n[watchdog] stopped.")
