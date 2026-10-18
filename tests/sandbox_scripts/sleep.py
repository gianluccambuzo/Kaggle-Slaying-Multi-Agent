import subprocess
import sys
import time
from pathlib import Path

# Optional first arg "spawn": launch a child that also sleeps, record pids.
if len(sys.argv) > 1 and sys.argv[1] == "spawn":
    child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
    Path("pids.txt").write_text(f"{child.pid}\n", encoding="utf-8")
time.sleep(60)
