import logging
import sys
from datetime import datetime

_LOG_STORE = []   # In-memory log buffer for the UI

class UIHandler(logging.Handler):
    def emit(self, record):
        _LOG_STORE.append({
            "time":    datetime.utcnow().strftime("%H:%M:%S"),
            "level":   record.levelname,
            "msg":     self.format(record),
        })
        if len(_LOG_STORE) > 500:
            _LOG_STORE.pop(0)

def get_logger(name="constellation"):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%H:%M:%S"))
        log.addHandler(sh)
        log.addHandler(UIHandler())
    return log

def get_ui_logs():
    return list(reversed(_LOG_STORE))   # newest first
