import os
import logging
from datetime import datetime
try:
    import structlog  # type: ignore
    _STRUCTLOG_AVAILABLE = True
except Exception:
    structlog = None
    _STRUCTLOG_AVAILABLE = False


class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        logger_name = os.path.basename(name)

        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[console_handler, file_handler]
        )

        if _STRUCTLOG_AVAILABLE:
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.EventRenamer(to="event"),
                    structlog.processors.JSONRenderer()
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            return structlog.get_logger(logger_name)
        else:
            return logging.getLogger(logger_name)

_GLOBAL_LOGGER_FACTORY = CustomLogger()
GLOBAL_LOGGER = _GLOBAL_LOGGER_FACTORY.get_logger(__name__)