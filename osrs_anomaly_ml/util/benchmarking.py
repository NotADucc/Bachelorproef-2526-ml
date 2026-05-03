import datetime
import functools
import inspect
import os
import time
from typing import Callable

from .log import get_logger

logger = get_logger(__name__)


def benchmark(callback: Callable):
    """ Decorator that benchmarks time and memory usage. """
    @functools.wraps(callback)
    async def wrapper(*args, **kwargs):
        filename = os.path.basename(inspect.getfile(callback))
        start_time = time.perf_counter()

        try:
            result = callback(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            end_time = time.perf_counter()

            logger.debug(
                f"[{filename}] - {callback.__name__} took {datetime.timedelta(seconds=end_time - start_time)}"
            )

    return wrapper
