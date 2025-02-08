import logging
import functools
import traceback


def exception_to_logs(logger: logging.Logger, custom_message: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录异常信息和 Traceback 数据
                if custom_message:
                    logger.error(
                        f"{custom_message} - Exception occurred in '{func.__name__}': {str(e)}\n{traceback.format_exc()}"
                    )
                else:
                    logger.error(
                        f"Exception occurred in {func.__name__}: {str(e)}]n{traceback.format_exc()}"
                    )
                raise

        return wrapper

    return decorator


def log_error(logger: logging.Logger, e: Exception, custom_message: str = ""):
    logger.error(f"{custom_message} - {str(e)}\n{traceback.format_exc()}")
