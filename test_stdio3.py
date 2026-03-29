"""Test that os.dup(1) survives setup_logging destroying the original stdout."""
import sys
import json
import os
import io

def log(msg):
    sys.stderr.write(f"[TEST] {msg}\n")
    sys.stderr.flush()

# Clone fd1 before anything
_IPC_STDOUT_FD = os.dup(1)
_IPC_STDIN_FD = os.dup(0)
_IPC_STDOUT = io.TextIOWrapper(
    io.BufferedWriter(io.FileIO(_IPC_STDOUT_FD, 'w', closefd=False)),
    encoding='utf-8', errors='replace', line_buffering=True,
)
_IPC_STDIN = io.TextIOWrapper(
    io.BufferedReader(io.FileIO(_IPC_STDIN_FD, 'r', closefd=False)),
    encoding='utf-8', errors='replace',
)

log("IPC streams created via os.dup()")

# Now import the logger which may mess up sys.stdout
from src.core.logger import get_logger, setup_logging
from src.core.db import init_db

log(f"After imports - sys.stdout closed: {sys.stdout.closed}")
log(f"After imports - IPC_STDOUT closed: {_IPC_STDOUT.closed}")

# Now re-init logging like main() does
import logging as _logging
from src.core import logger as _logmod
_logmod._initialized = False
_logmod._root_logger = None
setup_logging(log_to_console=False, log_to_file=True)

log(f"After setup_logging - sys.stdout closed: {sys.stdout.closed}")
log(f"After setup_logging - IPC_STDOUT closed: {_IPC_STDOUT.closed}")

# Try writing via the cloned fd
try:
    msg = json.dumps({"jsonrpc": "2.0", "method": "event.ready", "params": {"version": "4.0.0"}})
    _IPC_STDOUT.write(msg + "\n")
    _IPC_STDOUT.flush()
    log("IPC_STDOUT write OK!")
except Exception as e:
    log(f"IPC_STDOUT write FAILED: {e}")

# Try reading stdin via cloned fd  
log("Waiting for stdin...")
line = _IPC_STDIN.readline()
log(f"Got: {line.strip()}")

if line.strip():
    req = json.loads(line)
    resp = json.dumps({"jsonrpc": "2.0", "id": req.get("id"), "result": "pong"})
    _IPC_STDOUT.write(resp + "\n")
    _IPC_STDOUT.flush()
    log("Response sent OK!")
