import subprocess
import sys
import time
from pathlib import Path

# Leave a background child behind and exit normally.
child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(60)"])
Path("pids.txt").write_text(f"{child.pid}\n", encoding="utf-8")
time.sleep(1.0)  # long enough for the sandbox monitor to see the child
