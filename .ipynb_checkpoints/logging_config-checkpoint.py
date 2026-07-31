import logging
import json

logger = logging.getLogger("iris-api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()

class JsonFormatter(logging.Formatter):

    def format(self, record):
        return json.dumps({
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record)
        })

handler.setFormatter(JsonFormatter())

logger.addHandler(handler)