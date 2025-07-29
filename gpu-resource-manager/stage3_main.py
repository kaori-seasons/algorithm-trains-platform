import sys
import signal
import time
from typing import Optional
from logger import setup_logging, get_logger
from config import config
from k8s_client import K8sResourceManager
from gpu_parser import GPUResourceParser
from gpu_memory_guard import GPUMemoryGuard
from stage2_integration import Stage2ResourceManager
from stage3_integration import Stage3ResourceManager

logger = get_logger(__name__)

class Stage3Main:
    """第三阶段主程序"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None, prometheus_url: str = None):
        setup_logging()
        logger.info("初始化第三阶段资源管理器")
        
        # 初始化基础组件
        self.k8s_client = K8sResourceManager(kubeconfig_path)
        self.gpu_parser = GPUResourceParser(config.config)
        self.memory_guard = GPUMemoryGuard(self.k8s_client, self.gpu_parser)
        
        # 初始化第二阶段管理器
        prometheus_url = prometheus_url or "http://localhost:9090"
        self.stage2_manager = Stage2ResourceManager(
            self.k8s_client, 
            self.gpu_parser, 
            self.memory_guard, 
            prometheus_url
        )
        
        # 初始化第三阶段管理器
        self.stage3_manager = Stage3ResourceManager(
            self.k8s_client,
            self.stage2_manager,
            prometheus_url
        )
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def start(self):
        """启动第三阶段服务"""
        logger.info("启动第三阶段资源管理器")
        
        try:
            # 启动第二阶段服务
            self.stage2_manager.start_all_services()
            
            # 启动第三阶段服务
            self.stage3_manager.start_all_services()
            
            logger.info("第三阶段资源管理器已启动")
            
            # 保持运行
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        except Exception as e:
            logger.error(f"第三阶段服务运行错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止第三阶段服务"""
        logger.info("停止第三阶段资源管理器")
        
        # 停止第三阶段服务
        self.stage3_manager.stop_all_services()
        
        # 停止第二阶段服务
        self.stage2_manager.stop_all_services()
        
        logger.info("第三阶段资源管理器已停止")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，正在关闭...")
        self.stop()
        sys.exit(0)
    
    def get_status(self):
        """获取系统状态"""
        return {
            'stage2_status': self.stage2_manager.get_system_status(),
            'stage3_status': self.stage3_manager.get_monitoring_status(),
            'cluster_summary': self.stage3_manager.get_cluster_summary(),
            'performance_metrics': self.stage3_manager.get_performance_metrics()
        }

def main():
    """主函数"""
    kubeconfig_path = config.get('KUBECONFIG_PATH')
    prometheus_url = "http://localhost:9090"
    
    stage3_main = Stage3Main(kubeconfig_path, prometheus_url)
    stage3_main.start()

if __name__ == "__main__":
    main() 