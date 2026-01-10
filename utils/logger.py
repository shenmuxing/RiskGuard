import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback

class PrintToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, message):
        self.buffer.append(message)
        if message.endswith('\n'):
            self.flush()

    def flush(self):
        message = ''.join(self.buffer).strip()
        if message:
            self.logger.log(self.level, message)
        self.buffer = []

class PrintToLoggerError:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = []

    def write(self, message):
        self.buffer.append(message)
        if message.endswith('\n'):
            self.flush()

    def flush(self):
        message = ''.join(self.buffer).strip()
        if message:
            self.logger.error(message)
        self.buffer = []

def setup_logger(cfg):
    # 创建 logger
    logger = logging.getLogger('app_logger')
    logger.setLevel(cfg.log_level)

    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(cfg.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建 RotatingFileHandler
    file_handler = RotatingFileHandler(
        cfg.log_file,
        maxBytes=cfg.log_max_size,
        backupCount=cfg.log_backup_count
    )
    file_handler.setLevel(cfg.log_level)

    # 创建 StreamHandler 用于控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(cfg.log_level)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 重定向 stdout 和 stderr 到 logger
    sys.stdout = PrintToLogger(logger, logging.INFO)
    sys.stderr = PrintToLoggerError(logger)

    # 添加 shutdown 方法
    def shutdown():
        sys.stdout = sys.__stdout__  # 恢复原始的 stdout
        sys.stderr = sys.__stderr__  # 恢复原始的 stderr
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    logger.shutdown = shutdown

    return logger

# 全局异常处理器
def global_exception_handler(exctype, value, tb):
    logger = logging.getLogger('app_logger')
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    logger.error(f"Uncaught exception:\n{error_msg}")

# Test the logger setup
if __name__ == '__main__':
    class Config:
        log_level = logging.DEBUG
        log_file = 'logs/app.log'
        log_max_size = 1024 * 1024 * 5  # 5 MB
        log_backup_count = 3

    cfg = Config()
    logger = setup_logger(cfg)

    # 设置全局异常处理器
    sys.excepthook = global_exception_handler

    try:
        logger.info("This is an info message.")
        print("This is a print statement.")
        logger.error("This is an error message.")
        
        # 测试未捕获的异常
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.exception("Caught an exception:")
    finally:
        logger.shutdown()