import sys
import os
import datetime
import atexit
import logging

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"train_{timestamp}.log")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    f = open(logfile, "a", buffering=1, encoding="utf-8")
    atexit.register(f.close)
    class Tee:
        def __init__(self, *files):
            self._files = files
            self.encoding = getattr(files[0], "encoding", "utf-8")
        def write(self, data):
            if not data:
                return
            for fh in self._files:
                try:
                    fh.write(data)
                except Exception:
                    pass
        def flush(self):
            for fh in self._files:
                try:
                    fh.flush()
                except Exception:
                    pass
        def isatty(self):
            return any(getattr(fh, "isatty", lambda: False)() for fh in self._files)
    sys.stdout = Tee(orig_stdout, f)
    sys.stderr = Tee(orig_stderr, f)
    log_formatter = logging.Formatter(fmt="%(asctime)s — %(levelname)s — %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers): root_logger.removeHandler(h)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(log_formatter)
    sh = logging.StreamHandler(orig_stdout)
    sh.setFormatter(log_formatter)
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)
    return logfile
