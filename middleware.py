import time
from logging_config import logger

async def logging_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path} "
        f"{response.status_code} {duration}ms"
    )

    response.headers["X-Process-Time-ms"] = str(duration)

    return response