"""Quick test to see what happens with sys.stdout and sys.__stdout__ when spawned."""
import sys
import os
import json

# Write to stderr for diagnostics
def log(msg):
    sys.stderr.write(f"[TEST] {msg}\n")
    sys.stderr.flush()

log(f"sys.stdout: {sys.stdout}")
log(f"sys.__stdout__: {sys.__stdout__}")
log(f"sys.stdout is sys.__stdout__: {sys.stdout is sys.__stdout__}")
log(f"sys.stdout.fileno(): {sys.stdout.fileno()}")
try:
    log(f"sys.__stdout__.fileno(): {sys.__stdout__.fileno()}")
except Exception as e:
    log(f"sys.__stdout__.fileno() error: {e}")
log(f"sys.stdout.closed: {sys.stdout.closed}")
try:
    log(f"sys.__stdout__.closed: {sys.__stdout__.closed}")
except Exception as e:
    log(f"sys.__stdout__.closed error: {e}")

# Try writing via sys.stdout
try:
    msg = json.dumps({"test": "via sys.stdout"})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()
    log("sys.stdout write OK")
except Exception as e:
    log(f"sys.stdout write FAILED: {e}")

# Try writing via sys.__stdout__
try:
    msg = json.dumps({"test": "via sys.__stdout__"})
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()
    log("sys.__stdout__ write OK")
except Exception as e:
    log(f"sys.__stdout__ write FAILED: {e}")

# Try writing via os.fdopen
try:
    import io
    raw_stdout = io.FileIO(1, 'w', closefd=False)
    buffered = io.BufferedWriter(raw_stdout)
    text_stdout = io.TextIOWrapper(buffered, encoding='utf-8')
    msg = json.dumps({"test": "via fd1 direct"})
    text_stdout.write(msg + "\n")
    text_stdout.flush()
    log("fd1 direct write OK")
except Exception as e:
    log(f"fd1 direct write FAILED: {e}")

# Now simulate what logger.py does and test again
import io
try:
    console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    log(f"Created console_stream wrapper: {console_stream}")
    
    # After this, try sys.stdout again
    msg2 = json.dumps({"test": "after wrapper via sys.stdout"})
    sys.stdout.write(msg2 + "\n")
    sys.stdout.flush()
    log("sys.stdout write after wrapper OK")
except Exception as e:
    log(f"After wrapper error: {e}")

# Read one line from stdin
log("Waiting for stdin...")
line = sys.stdin.readline()
log(f"Got from stdin: {line.strip()}")
try:
    req = json.loads(line)
    resp = json.dumps({"result": "echo", "input": req})
    sys.stdout.write(resp + "\n")
    sys.stdout.flush()
    log("Echo response sent OK")
except Exception as e:
    log(f"Echo failed: {e}")
