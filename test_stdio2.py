"""Test what happens with stdout after importing the IPC server module."""
import sys
import json

def log(msg):
    sys.stderr.write(f"[TEST] {msg}\n")
    sys.stderr.flush()

log(f"BEFORE imports - sys.stdout: {sys.stdout}, closed={sys.stdout.closed}")
log(f"BEFORE imports - sys.__stdout__: {sys.__stdout__}, closed={sys.__stdout__.closed}")

# Import the IPC modules like server.py does
from src.core.logger import get_logger, setup_logging
from src.core.db import init_db

log(f"AFTER imports - sys.stdout: {sys.stdout}, closed={sys.stdout.closed}")
log(f"AFTER imports - sys.__stdout__: {sys.__stdout__}, closed={sys.__stdout__.closed}")

# Test writing
try:
    msg = json.dumps({"test": "after_imports_stdout"})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    log("sys.stdout write after imports OK")
except Exception as e:
    log(f"sys.stdout write after imports FAILED: {e}")

try:
    msg = json.dumps({"test": "after_imports___stdout__"})
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()
    log("sys.__stdout__ write after imports OK")
except Exception as e:
    log(f"sys.__stdout__ write after imports FAILED: {e}")

# Now simulate what main() does
log("--- Now calling setup_logging(log_to_console=False, log_to_file=True) ---")
import logging as _logging
from src.core import logger as _logmod
_logmod._initialized = False
_logmod._root_logger = None
setup_logging(log_to_console=False, log_to_file=True)

log(f"AFTER setup_logging - sys.stdout: {sys.stdout}, closed={sys.stdout.closed}")
log(f"AFTER setup_logging - sys.__stdout__: {sys.__stdout__}, closed={sys.__stdout__.closed}")

# Test writing again
try:
    msg = json.dumps({"test": "after_setup_logging_stdout"})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    log("sys.stdout write after setup_logging OK")
except Exception as e:
    log(f"sys.stdout write after setup_logging FAILED: {e}")

try:
    msg = json.dumps({"test": "after_setup_logging___stdout__"})
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()
    log("sys.__stdout__ write after setup_logging OK")
except Exception as e:
    log(f"sys.__stdout__ write after setup_logging FAILED: {e}")

# Add stderr handler like main() does
root = _logging.getLogger('epicbot')
stderr_handler = _logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(_logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))
stderr_handler.setLevel(_logging.INFO)
root.addHandler(stderr_handler)

log(f"AFTER adding stderr handler - sys.stdout: {sys.stdout}, closed={sys.stdout.closed}")
log(f"AFTER adding stderr handler - sys.__stdout__: {sys.__stdout__}, closed={sys.__stdout__.closed}")

# Now init DB and test
init_db()

log(f"AFTER init_db - sys.stdout: {sys.stdout}, closed={sys.stdout.closed}")
log(f"AFTER init_db - sys.__stdout__: {sys.__stdout__}, closed={sys.__stdout__.closed}")

# Final test
try:
    msg = json.dumps({"test": "final_stdout"})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    log("sys.stdout final write OK")
except Exception as e:
    log(f"sys.stdout final write FAILED: {e}")

try:
    msg = json.dumps({"test": "final___stdout__"})
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()
    log("sys.__stdout__ final write OK")
except Exception as e:
    log(f"sys.__stdout__ final write FAILED: {e}")

# Test stdin
log("Waiting for stdin line...")
line = sys.stdin.readline()
log(f"Got: {line.strip()}")
if line.strip():
    resp = json.dumps({"result": "ok", "echo": json.loads(line)})
    sys.__stdout__.write(resp + "\n")
    sys.__stdout__.flush()
    log("Response sent via sys.__stdout__")
