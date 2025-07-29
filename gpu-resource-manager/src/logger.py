import logging
import sys
from .config import config

def setup_logging():
    """设置日志配置"""
    log_level = getattr(logging, config.get('LOG_LEVEL', 'INFO').upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('resource_manager.log')
        ]
    )
    
    # 设置第三方库日志级别
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

# 获取logger实例
def get_logger(name: str) -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name) 