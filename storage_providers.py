# 存储提供者实现
import os
import json
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import subprocess
import tempfile
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageProvider(ABC):
    """
    存储提供者抽象基类
    """
    
    @abstractmethod
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载存储
        """
        pass
    
    @abstractmethod
    def unmount(self, mount_path: str) -> bool:
        """
        卸载存储
        """
        pass
    
    @abstractmethod
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取存储信息
        """
        pass
    
    @abstractmethod
    def check_health(self) -> bool:
        """
        检查存储健康状态
        """
        pass


class PVCProvider(StorageProvider):
    """
    Kubernetes PVC存储提供者
    支持动态创建、挂载和管理PVC
    """
    
    def __init__(self, kube_config_path: Optional[str] = None):
        """
        初始化PVC提供者
        
        Args:
            kube_config_path: Kubernetes配置文件路径，如果为None则使用默认配置
        """
        self.kube_config_path = kube_config_path
        self.core_v1_api = None
        self.storage_v1_api = None
        self.apps_v1_api = None
        self._init_kubernetes_client()
        
        # 存储挂载信息
        self.mount_info = {}
        
        # 挂载模式：pod_mount（Pod挂载）或 host_mount（宿主机挂载）
        self.mount_mode = 'pod_mount'
        
    def _init_kubernetes_client(self):
        """
        初始化Kubernetes客户端
        """
        try:
            if self.kube_config_path and os.path.exists(self.kube_config_path):
                config.load_kube_config(config_file=self.kube_config_path)
            else:
                # 尝试加载集群内配置，如果失败则尝试默认配置
                try:
                    config.load_incluster_config()  # 在集群内运行时使用
                except Exception:
                    # 如果不在集群内，尝试加载默认配置
                    try:
                        config.load_kube_config()  # 尝试加载默认kubeconfig
                    except Exception:
                        logger.warning("无法加载Kubernetes配置，PVC功能将不可用")
                        self.core_v1_api = None
                        self.storage_v1_api = None
                        self.apps_v1_api = None
                        return
            
            self.core_v1_api = client.CoreV1Api()
            self.storage_v1_api = client.StorageV1Api()
            self.apps_v1_api = client.AppsV1Api()
            logger.info("Kubernetes客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Kubernetes客户端初始化失败: {e}")
            self.core_v1_api = None
            self.storage_v1_api = None
            self.apps_v1_api = None
    
    def set_mount_mode(self, mode: str):
        """
        设置挂载模式
        
        Args:
            mode: 'pod_mount' 或 'host_mount'
        """
        if mode not in ['pod_mount', 'host_mount']:
            raise ValueError("挂载模式必须是 'pod_mount' 或 'host_mount'")
        
        self.mount_mode = mode
        logger.info(f"设置挂载模式为: {mode}")
    
    def create_pvc(self, pvc_config: Dict[str, Any]) -> str:
        """
        创建PVC
        
        Args:
            pvc_config: PVC配置
                {
                    'name': 'pvc-name',
                    'namespace': 'default',
                    'storage_class': 'standard',
                    'access_modes': ['ReadWriteOnce'],
                    'storage_size': '10Gi',
                    'labels': {'app': 'train-platform'},
                    'annotations': {'description': 'Training data storage'}
                }
        
        Returns:
            PVC名称
        """
        try:
            # 构建PVC对象
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=pvc_config['name'],
                    namespace=pvc_config.get('namespace', 'default'),
                    labels=pvc_config.get('labels', {}),
                    annotations=pvc_config.get('annotations', {})
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=pvc_config['access_modes'],
                    resources=client.V1ResourceRequirements(
                        requests={'storage': pvc_config['storage_size']}
                    ),
                    storage_class_name=pvc_config.get('storage_class')
                )
            )
            
            # 创建PVC
            namespace = pvc_config.get('namespace', 'default')
            created_pvc = self.core_v1_api.create_namespaced_persistent_volume_claim(
                namespace=namespace,
                body=pvc
            )
            
            logger.info(f"PVC {pvc_config['name']} 创建成功")
            return created_pvc.metadata.name
            
        except ApiException as e:
            if e.status == 409:  # 已存在
                logger.warning(f"PVC {pvc_config['name']} 已存在")
                return pvc_config['name']
            else:
                logger.error(f"创建PVC失败: {e}")
                raise
        except Exception as e:
            logger.error(f"创建PVC时发生错误: {e}")
            raise
    
    def delete_pvc(self, pvc_name: str, namespace: str = 'default') -> bool:
        """
        删除PVC
        
        Args:
            pvc_name: PVC名称
            namespace: 命名空间
        
        Returns:
            是否删除成功
        """
        try:
            self.core_v1_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=namespace
            )
            logger.info(f"PVC {pvc_name} 删除成功")
            return True
            
        except ApiException as e:
            if e.status == 404:  # 不存在
                logger.warning(f"PVC {pvc_name} 不存在")
                return True
            else:
                logger.error(f"删除PVC失败: {e}")
                return False
        except Exception as e:
            logger.error(f"删除PVC时发生错误: {e}")
            return False
    
    def get_pvc_status(self, pvc_name: str, namespace: str = 'default') -> Dict[str, Any]:
        """
        获取PVC状态
        
        Args:
            pvc_name: PVC名称
            namespace: 命名空间
        
        Returns:
            PVC状态信息
        """
        try:
            pvc = self.core_v1_api.read_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=namespace
            )
            
            return {
                'name': pvc.metadata.name,
                'namespace': pvc.metadata.namespace,
                'phase': pvc.status.phase,
                'access_modes': pvc.status.access_modes,
                'capacity': pvc.status.capacity,
                'volume_name': pvc.spec.volume_name,
                'storage_class': pvc.spec.storage_class_name,
                'labels': pvc.metadata.labels,
                'annotations': pvc.metadata.annotations
            }
            
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"PVC {pvc_name} 不存在")
                return {}
            else:
                logger.error(f"获取PVC状态失败: {e}")
                raise
        except Exception as e:
            logger.error(f"获取PVC状态时发生错误: {e}")
            raise
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载PVC存储
        
        Args:
            mount_config: 挂载配置
                {
                    'pvc_name': 'training-data-pvc',
                    'namespace': 'default',
                    'mount_path': '/mnt/training-data',
                    'sub_path': 'user-data',  # 可选，子路径
                    'read_only': False,  # 可选，是否只读
                    'create_if_not_exists': True,  # 可选，不存在时是否创建
                    'mount_mode': 'pod_mount'  # 可选，挂载模式
                }
        
        Returns:
            挂载结果信息
        """
        try:
            pvc_name = mount_config['pvc_name']
            namespace = mount_config.get('namespace', 'default')
            mount_path = mount_config['mount_path']
            sub_path = mount_config.get('sub_path')
            read_only = mount_config.get('read_only', False)
            create_if_not_exists = mount_config.get('create_if_not_exists', True)
            mount_mode = mount_config.get('mount_mode', self.mount_mode)
            
            # 检查PVC是否存在
            pvc_status = self.get_pvc_status(pvc_name, namespace)
            
            if not pvc_status and create_if_not_exists:
                # 创建PVC
                pvc_config = {
                    'name': pvc_name,
                    'namespace': namespace,
                    'storage_class': mount_config.get('storage_class', 'standard'),
                    'access_modes': mount_config.get('access_modes', ['ReadWriteOnce']),
                    'storage_size': mount_config.get('storage_size', '10Gi'),
                    'labels': mount_config.get('labels', {}),
                    'annotations': mount_config.get('annotations', {})
                }
                self.create_pvc(pvc_config)
                pvc_status = self.get_pvc_status(pvc_name, namespace)
            
            if not pvc_status:
                raise ValueError(f"PVC {pvc_name} 不存在且无法创建")
            
            # 检查PVC是否已绑定
            if pvc_status['phase'] != 'Bound':
                raise ValueError(f"PVC {pvc_name} 未绑定，当前状态: {pvc_status['phase']}")
            
            # 根据挂载模式选择挂载方法
            if mount_mode == 'pod_mount':
                return self._mount_in_pod(mount_config, pvc_status)
            elif mount_mode == 'host_mount':
                return self._mount_on_host(mount_config, pvc_status)
            else:
                raise ValueError(f"不支持的挂载模式: {mount_mode}")
            
        except Exception as e:
            logger.error(f"挂载PVC失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_in_pod(self, mount_config: Dict[str, Any], pvc_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        在Pod中挂载PVC（推荐方式）
        """
        try:
            mount_path = mount_config['mount_path']
            pvc_name = mount_config['pvc_name']
            namespace = mount_config.get('namespace', 'default')
            sub_path = mount_config.get('sub_path')
            read_only = mount_config.get('read_only', False)
            
            # 创建挂载点目录
            os.makedirs(mount_path, exist_ok=True)
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': pvc_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'sub_path': sub_path,
                'read_only': read_only,
                'volume_name': pvc_status.get('volume_name'),
                'access_modes': pvc_status.get('access_modes', []),
                'capacity': pvc_status.get('capacity', {}),
                'mount_mode': 'pod_mount',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            logger.info(f"PVC {pvc_name} 在Pod中挂载到 {mount_path} 成功")
            
            return {
                'success': True,
                'mount_path': mount_path,
                'pvc_name': pvc_name,
                'volume_name': pvc_status.get('volume_name'),
                'mount_mode': 'pod_mount',
                'mount_info': mount_info
            }
            
        except Exception as e:
            logger.error(f"Pod挂载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_on_host(self, mount_config: Dict[str, Any], pvc_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        在宿主机上挂载PVC（需要特殊权限）
        """
        try:
            mount_path = mount_config['mount_path']
            pvc_name = mount_config['pvc_name']
            namespace = mount_config.get('namespace', 'default')
            
            # 获取PV信息
            volume_name = pvc_status.get('volume_name')
            if not volume_name:
                raise ValueError("PVC未绑定到PV")
            
            # 获取PV详情
            pv = self.core_v1_api.read_persistent_volume(volume_name)
            
            # 根据PV类型执行不同的挂载策略
            if pv.spec.host_path:
                return self._mount_host_path_pv(pv, mount_path, mount_config)
            elif pv.spec.nfs:
                return self._mount_nfs_pv(pv, mount_path, mount_config)
            elif pv.spec.aws_elastic_block_store:
                return self._mount_ebs_pv(pv, mount_path, mount_config)
            else:
                # 对于其他类型的PV，尝试使用kubectl命令挂载
                return self._mount_with_kubectl(pvc_name, namespace, mount_path, mount_config)
            
        except Exception as e:
            logger.error(f"宿主机挂载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_host_path_pv(self, pv, mount_path: str, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载hostPath类型的PV
        """
        try:
            host_path = pv.spec.host_path.path
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 创建符号链接
            if os.path.exists(mount_path):
                if os.path.islink(mount_path):
                    os.unlink(mount_path)
                elif os.path.isdir(mount_path) and not os.listdir(mount_path):
                    os.rmdir(mount_path)
                else:
                    raise Exception(f"挂载点 {mount_path} 已存在且不为空")
            
            os.symlink(host_path, mount_path)
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': mount_config['pvc_name'],
                'namespace': mount_config.get('namespace', 'default'),
                'mount_path': mount_path,
                'host_path': host_path,
                'mount_mode': 'host_mount',
                'pv_type': 'host_path',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'host_path': host_path,
                'mount_mode': 'host_mount'
            }
            
        except Exception as e:
            logger.error(f"挂载hostPath PV失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_nfs_pv(self, pv, mount_path: str, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载NFS类型的PV
        """
        try:
            nfs_server = pv.spec.nfs.server
            nfs_path = pv.spec.nfs.path
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 执行mount命令
            mount_cmd = [
                'mount', '-t', 'nfs',
                '-o', 'rw,sync',
                f"{nfs_server}:{nfs_path}",
                mount_path
            ]
            
            result = subprocess.run(mount_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"NFS挂载失败: {result.stderr}")
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': mount_config['pvc_name'],
                'namespace': mount_config.get('namespace', 'default'),
                'mount_path': mount_path,
                'nfs_server': nfs_server,
                'nfs_path': nfs_path,
                'mount_mode': 'host_mount',
                'pv_type': 'nfs',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'nfs_server': nfs_server,
                'mount_mode': 'host_mount'
            }
            
        except Exception as e:
            logger.error(f"挂载NFS PV失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_ebs_pv(self, pv, mount_path: str, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载EBS类型的PV（AWS）
        """
        try:
            # 获取EBS卷ID
            volume_id = pv.spec.aws_elastic_block_store.volume_id
            
            # 检查设备是否已挂载
            device_path = f"/dev/xvdf"  # 假设的设备路径
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 格式化设备（如果需要）
            # 这里需要根据实际情况处理
            
            # 挂载设备
            mount_cmd = ['mount', device_path, mount_path]
            result = subprocess.run(mount_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"EBS挂载失败: {result.stderr}")
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': mount_config['pvc_name'],
                'namespace': mount_config.get('namespace', 'default'),
                'mount_path': mount_path,
                'volume_id': volume_id,
                'device_path': device_path,
                'mount_mode': 'host_mount',
                'pv_type': 'aws_ebs',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'volume_id': volume_id,
                'mount_mode': 'host_mount'
            }
            
        except Exception as e:
            logger.error(f"挂载EBS PV失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _mount_with_kubectl(self, pvc_name: str, namespace: str, mount_path: str, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用kubectl命令挂载PVC
        """
        try:
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 使用kubectl cp命令复制数据（临时方案）
            # 注意：这不是真正的挂载，而是数据复制
            temp_pod_name = f"temp-mount-{pvc_name}"
            
            # 创建临时Pod来访问PVC
            temp_pod = self._create_temp_pod(pvc_name, namespace, temp_pod_name)
            
            # 等待Pod运行
            self._wait_for_pod_ready(temp_pod_name, namespace)
            
            # 复制数据到本地
            kubectl_cp_cmd = [
                'kubectl', 'cp',
                f"{namespace}/{temp_pod_name}:/data",
                mount_path
            ]
            
            result = subprocess.run(kubectl_cp_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"kubectl cp失败: {result.stderr}")
            
            # 删除临时Pod
            self._delete_temp_pod(temp_pod_name, namespace)
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': pvc_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'mount_mode': 'host_mount',
                'pv_type': 'kubectl_copy',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'pvc_name': pvc_name,
                'mount_mode': 'host_mount'
            }
            
        except Exception as e:
            logger.error(f"kubectl挂载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_temp_pod(self, pvc_name: str, namespace: str, pod_name: str) -> client.V1Pod:
        """
        创建临时Pod来访问PVC
        """
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace=namespace
            ),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name="temp-container",
                        image="busybox:latest",
                        command=["sleep", "3600"],
                        volume_mounts=[
                            client.V1VolumeMount(
                                name="pvc-volume",
                                mount_path="/data"
                            )
                        ]
                    )
                ],
                volumes=[
                    client.V1Volume(
                        name="pvc-volume",
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=pvc_name
                        )
                    )
                ],
                restart_policy="Never"
            )
        )
        
        return self.core_v1_api.create_namespaced_pod(
            namespace=namespace,
            body=pod
        )
    
    def _wait_for_pod_ready(self, pod_name: str, namespace: str, timeout: int = 300):
        """
        等待Pod就绪
        """
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                pod = self.core_v1_api.read_namespaced_pod(pod_name, namespace)
                if pod.status.phase == 'Running':
                    return True
                elif pod.status.phase in ['Failed', 'Succeeded']:
                    return False
            except Exception:
                pass
            
            time.sleep(5)
        
        raise Exception(f"Pod {pod_name} 在 {timeout} 秒内未就绪")
    
    def _delete_temp_pod(self, pod_name: str, namespace: str):
        """
        删除临时Pod
        """
        try:
            self.core_v1_api.delete_namespaced_pod(pod_name, namespace)
        except Exception as e:
            logger.warning(f"删除临时Pod失败: {e}")
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载PVC存储
        """
        try:
            if mount_path not in self.mount_info:
                logger.warning(f"挂载路径 {mount_path} 未找到")
                return True
            
            mount_info = self.mount_info[mount_path]
            mount_mode = mount_info.get('mount_mode', 'pod_mount')
            
            if mount_mode == 'host_mount':
                return self._unmount_from_host(mount_path, mount_info)
            else:
                # Pod挂载模式，只需要清理记录
                del self.mount_info[mount_path]
                return True
            
        except Exception as e:
            logger.error(f"卸载PVC失败: {e}")
            return False
    
    def _unmount_from_host(self, mount_path: str, mount_info: Dict[str, Any]) -> bool:
        """
        从宿主机卸载
        """
        try:
            pv_type = mount_info.get('pv_type')
            
            if pv_type == 'host_path':
                # 删除符号链接
                if os.path.islink(mount_path):
                    os.unlink(mount_path)
            elif pv_type == 'nfs':
                # 卸载NFS
                umount_cmd = ['umount', mount_path]
                result = subprocess.run(umount_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"NFS卸载失败: {result.stderr}")
                    return False
            elif pv_type == 'aws_ebs':
                # 卸载EBS
                umount_cmd = ['umount', mount_path]
                result = subprocess.run(umount_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"EBS卸载失败: {result.stderr}")
                    return False
            elif pv_type == 'kubectl_copy':
                # 删除复制的数据
                if os.path.exists(mount_path):
                    shutil.rmtree(mount_path, ignore_errors=True)
            
            # 从记录中删除
            del self.mount_info[mount_path]
            return True
            
        except Exception as e:
            logger.error(f"宿主机卸载失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取存储信息
        
        Args:
            mount_path: 挂载路径
        
        Returns:
            存储信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 获取最新的PVC状态
        try:
            pvc_status = self.get_pvc_status(
                mount_info['pvc_name'], 
                mount_info['namespace']
            )
            
            # 合并信息
            info = mount_info.copy()
            info.update(pvc_status)
            
            # 添加文件系统信息
            if os.path.exists(mount_path):
                statvfs = os.statvfs(mount_path)
                info['filesystem'] = {
                    'total_space': statvfs.f_blocks * statvfs.f_frsize,
                    'free_space': statvfs.f_bavail * statvfs.f_frsize,
                    'used_space': (statvfs.f_blocks - statvfs.f_bavail) * statvfs.f_frsize,
                    'block_size': statvfs.f_frsize
                }
            
            return info
            
        except Exception as e:
            logger.error(f"获取存储信息失败: {e}")
            return mount_info
    
    def check_health(self) -> bool:
        """
        检查存储健康状态
        
        Returns:
            是否健康
        """
        try:
            # 检查Kubernetes API连接
            self.core_v1_api.list_namespace()
            
            # 检查所有挂载的PVC状态
            for mount_path, mount_info in self.mount_info.items():
                pvc_status = self.get_pvc_status(
                    mount_info['pvc_name'], 
                    mount_info['namespace']
                )
                
                if not pvc_status or pvc_status['phase'] != 'Bound':
                    logger.warning(f"PVC {mount_info['pvc_name']} 状态异常")
                    return False
                
                # 检查挂载点是否可访问
                if not os.path.exists(mount_path):
                    logger.warning(f"挂载点 {mount_path} 不存在")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def list_pvcs(self, namespace: str = 'default') -> List[Dict[str, Any]]:
        """
        列出命名空间中的所有PVC
        
        Args:
            namespace: 命名空间
        
        Returns:
            PVC列表
        """
        try:
            pvcs = self.core_v1_api.list_namespaced_persistent_volume_claim(
                namespace=namespace
            )
            
            pvc_list = []
            for pvc in pvcs.items:
                pvc_info = {
                    'name': pvc.metadata.name,
                    'namespace': pvc.metadata.namespace,
                    'phase': pvc.status.phase,
                    'access_modes': pvc.status.access_modes,
                    'capacity': pvc.status.capacity,
                    'volume_name': pvc.spec.volume_name,
                    'storage_class': pvc.spec.storage_class_name,
                    'labels': pvc.metadata.labels,
                    'annotations': pvc.metadata.annotations,
                    'creation_timestamp': pvc.metadata.creation_timestamp
                }
                pvc_list.append(pvc_info)
            
            return pvc_list
            
        except Exception as e:
            logger.error(f"列出PVC失败: {e}")
            return []
    
    def resize_pvc(self, pvc_name: str, new_size: str, namespace: str = 'default') -> bool:
        """
        调整PVC大小
        
        Args:
            pvc_name: PVC名称
            new_size: 新大小 (如 '20Gi')
            namespace: 命名空间
        
        Returns:
            是否调整成功
        """
        try:
            # 获取当前PVC
            pvc = self.core_v1_api.read_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=namespace
            )
            
            # 更新存储大小
            pvc.spec.resources.requests['storage'] = new_size
            
            # 应用更新
            self.core_v1_api.patch_namespaced_persistent_volume_claim(
                name=pvc_name,
                namespace=namespace,
                body=pvc
            )
            
            logger.info(f"PVC {pvc_name} 大小调整为 {new_size} 成功")
            return True
            
        except Exception as e:
            logger.error(f"调整PVC大小失败: {e}")
            return False
    
    def _is_path_in_use(self, path: str) -> bool:
        """
        检查路径是否正在被使用
        
        Args:
            path: 路径
        
        Returns:
            是否正在被使用
        """
        try:
            # 使用lsof检查是否有进程在使用该路径
            result = subprocess.run(
                ['lsof', path], 
                capture_output=True, 
                text=True
            )
            return result.returncode == 0 and result.stdout.strip() != ''
        except Exception:
            # 如果lsof不可用，使用简单的文件检查
            return os.path.exists(path) and os.listdir(path)
    
    def create_snapshot(self, pvc_name: str, snapshot_name: str, namespace: str = 'default') -> bool:
        """
        创建PVC快照
        
        Args:
            pvc_name: PVC名称
            snapshot_name: 快照名称
            namespace: 命名空间
        
        Returns:
            是否创建成功
        """
        try:
            # 检查是否支持快照
            try:
                self.storage_v1_api.list_volume_snapshot_class()
            except ApiException:
                logger.warning("当前集群不支持VolumeSnapshot")
                return False
            
            # 创建快照
            snapshot = client.V1VolumeSnapshot(
                metadata=client.V1ObjectMeta(
                    name=snapshot_name,
                    namespace=namespace
                ),
                spec=client.V1VolumeSnapshotSpec(
                    source=client.V1TypedLocalObjectReference(
                        kind="PersistentVolumeClaim",
                        name=pvc_name
                    )
                )
            )
            
            # 应用快照
            self.storage_v1_api.create_namespaced_volume_snapshot(
                namespace=namespace,
                body=snapshot
            )
            
            logger.info(f"PVC {pvc_name} 快照 {snapshot_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建快照失败: {e}")
            return False


class HostPathProvider(StorageProvider):
    """
    主机路径存储提供者
    """
    
    def __init__(self):
        self.mount_info = {}
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载主机路径
        """
        try:
            host_path = mount_config['host_path']
            mount_path = mount_config['mount_path']
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 创建符号链接或复制数据
            if mount_config.get('use_symlink', True):
                # 如果挂载点已存在且是符号链接，先删除
                if os.path.exists(mount_path) and os.path.islink(mount_path):
                    os.unlink(mount_path)
                elif os.path.exists(mount_path) and os.path.isdir(mount_path):
                    # 如果是目录且为空，删除它
                    if not os.listdir(mount_path):
                        os.rmdir(mount_path)
                    else:
                        # 目录不为空，使用复制模式
                        mount_config['use_symlink'] = False
                
                # 创建符号链接
                if not os.path.exists(mount_path):
                    os.symlink(host_path, mount_path)
            else:
                # 复制数据
                if os.path.exists(host_path):
                    shutil.copytree(host_path, mount_path, dirs_exist_ok=True)
            
            mount_info = {
                'host_path': host_path,
                'mount_path': mount_path,
                'use_symlink': mount_config.get('use_symlink', True),
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'host_path': host_path
            }
            
        except Exception as e:
            logger.error(f"挂载主机路径失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载主机路径
        """
        try:
            if mount_path in self.mount_info:
                mount_info = self.mount_info[mount_path]
                
                if mount_info.get('use_symlink'):
                    if os.path.islink(mount_path):
                        os.unlink(mount_path)
                else:
                    # 删除复制的数据
                    shutil.rmtree(mount_path, ignore_errors=True)
                
                del self.mount_info[mount_path]
            
            return True
            
        except Exception as e:
            logger.error(f"卸载主机路径失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取主机路径信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 添加文件系统信息
        if os.path.exists(mount_path):
            statvfs = os.statvfs(mount_path)
            mount_info['filesystem'] = {
                'total_space': statvfs.f_blocks * statvfs.f_frsize,
                'free_space': statvfs.f_bavail * statvfs.f_frsize,
                'used_space': (statvfs.f_blocks - statvfs.f_bavail) * statvfs.f_frsize
            }
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查主机路径健康状态
        """
        try:
            for mount_path, mount_info in self.mount_info.items():
                if not os.path.exists(mount_path):
                    return False
                if not os.path.exists(mount_info['host_path']):
                    return False
            return True
        except Exception:
            return False


class ConfigMapProvider(StorageProvider):
    """
    ConfigMap存储提供者
    """
    
    def __init__(self, kube_config_path: Optional[str] = None):
        self.kube_config_path = kube_config_path
        self.core_v1_api = None
        self._init_kubernetes_client()
        self.mount_info = {}
    
    def _init_kubernetes_client(self):
        """
        初始化Kubernetes客户端
        """
        try:
            if self.kube_config_path and os.path.exists(self.kube_config_path):
                config.load_kube_config(config_file=self.kube_config_path)
            else:
                # 尝试加载集群内配置，如果失败则尝试默认配置
                try:
                    config.load_incluster_config()
                except Exception:
                    try:
                        config.load_kube_config()
                    except Exception:
                        logger.warning("无法加载Kubernetes配置，ConfigMap功能将不可用")
                        self.core_v1_api = None
                        return
            
            self.core_v1_api = client.CoreV1Api()
            logger.info("Kubernetes客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Kubernetes客户端初始化失败: {e}")
            self.core_v1_api = None
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载ConfigMap
        """
        try:
            configmap_name = mount_config['configmap_name']
            namespace = mount_config.get('namespace', 'default')
            mount_path = mount_config['mount_path']
            
            # 获取ConfigMap
            configmap = self.core_v1_api.read_namespaced_config_map(
                name=configmap_name,
                namespace=namespace
            )
            
            # 创建挂载目录
            os.makedirs(mount_path, exist_ok=True)
            
            # 写入ConfigMap数据
            for key, value in configmap.data.items():
                file_path = os.path.join(mount_path, key)
                with open(file_path, 'w') as f:
                    f.write(value)
            
            mount_info = {
                'configmap_name': configmap_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'data_keys': list(configmap.data.keys()),
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'configmap_name': configmap_name
            }
            
        except Exception as e:
            logger.error(f"挂载ConfigMap失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载ConfigMap
        """
        try:
            if mount_path in self.mount_info:
                # 删除挂载的文件
                shutil.rmtree(mount_path, ignore_errors=True)
                del self.mount_info[mount_path]
            return True
        except Exception as e:
            logger.error(f"卸载ConfigMap失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取ConfigMap信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 获取最新的ConfigMap信息
        try:
            configmap = self.core_v1_api.read_namespaced_config_map(
                name=mount_info['configmap_name'],
                namespace=mount_info['namespace']
            )
            
            mount_info['data'] = configmap.data
            mount_info['metadata'] = {
                'creation_timestamp': configmap.metadata.creation_timestamp,
                'labels': configmap.metadata.labels,
                'annotations': configmap.metadata.annotations
            }
            
        except Exception as e:
            logger.error(f"获取ConfigMap信息失败: {e}")
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查ConfigMap健康状态
        """
        try:
            for mount_path, mount_info in self.mount_info.items():
                configmap = self.core_v1_api.read_namespaced_config_map(
                    name=mount_info['configmap_name'],
                    namespace=mount_info['namespace']
                )
                if not configmap:
                    return False
            return True
        except Exception:
            return False


class SecretProvider(StorageProvider):
    """
    Secret存储提供者
    """
    
    def __init__(self, kube_config_path: Optional[str] = None):
        self.kube_config_path = kube_config_path
        self.core_v1_api = None
        self._init_kubernetes_client()
        self.mount_info = {}
    
    def _init_kubernetes_client(self):
        """
        初始化Kubernetes客户端
        """
        try:
            if self.kube_config_path and os.path.exists(self.kube_config_path):
                config.load_kube_config(config_file=self.kube_config_path)
            else:
                # 尝试加载集群内配置，如果失败则尝试默认配置
                try:
                    config.load_incluster_config()
                except Exception:
                    try:
                        config.load_kube_config()
                    except Exception:
                        logger.warning("无法加载Kubernetes配置，Secret功能将不可用")
                        self.core_v1_api = None
                        return
            
            self.core_v1_api = client.CoreV1Api()
            logger.info("Kubernetes客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Kubernetes客户端初始化失败: {e}")
            self.core_v1_api = None
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载Secret
        """
        try:
            secret_name = mount_config['secret_name']
            namespace = mount_config.get('namespace', 'default')
            mount_path = mount_config['mount_path']
            
            # 获取Secret
            secret = self.core_v1_api.read_namespaced_secret(
                name=secret_name,
                namespace=namespace
            )
            
            # 创建挂载目录
            os.makedirs(mount_path, exist_ok=True)
            
            # 写入Secret数据
            for key, value in secret.data.items():
                file_path = os.path.join(mount_path, key)
                # Secret数据是base64编码的
                import base64
                decoded_value = base64.b64decode(value).decode('utf-8')
                with open(file_path, 'w') as f:
                    f.write(decoded_value)
                
                # 设置文件权限为600（只有所有者可读写）
                os.chmod(file_path, 0o600)
            
            mount_info = {
                'secret_name': secret_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'data_keys': list(secret.data.keys()),
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'secret_name': secret_name
            }
            
        except Exception as e:
            logger.error(f"挂载Secret失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载Secret
        """
        try:
            if mount_path in self.mount_info:
                # 删除挂载的文件
                shutil.rmtree(mount_path, ignore_errors=True)
                del self.mount_info[mount_path]
            return True
        except Exception as e:
            logger.error(f"卸载Secret失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取Secret信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 获取最新的Secret信息（不包含敏感数据）
        try:
            secret = self.core_v1_api.read_namespaced_secret(
                name=mount_info['secret_name'],
                namespace=mount_info['namespace']
            )
            
            mount_info['metadata'] = {
                'creation_timestamp': secret.metadata.creation_timestamp,
                'labels': secret.metadata.labels,
                'annotations': secret.metadata.annotations,
                'type': secret.type
            }
            
        except Exception as e:
            logger.error(f"获取Secret信息失败: {e}")
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查Secret健康状态
        """
        try:
            for mount_path, mount_info in self.mount_info.items():
                secret = self.core_v1_api.read_namespaced_secret(
                    name=mount_info['secret_name'],
                    namespace=mount_info['namespace']
                )
                if not secret:
                    return False
            return True
        except Exception:
            return False


class MemoryProvider(StorageProvider):
    """
    内存存储提供者
    """
    
    def __init__(self):
        self.mount_info = {}
        self.memory_storage = {}
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载内存存储
        """
        try:
            mount_path = mount_config['mount_path']
            max_size = mount_config.get('max_size', '1Gi')  # 默认1GB
            
            # 创建内存存储区域
            self.memory_storage[mount_path] = {
                'data': {},
                'max_size': self._parse_size(max_size),
                'used_size': 0
            }
            
            mount_info = {
                'mount_path': mount_path,
                'max_size': max_size,
                'type': 'memory',
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'max_size': max_size
            }
            
        except Exception as e:
            logger.error(f"挂载内存存储失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载内存存储
        """
        try:
            if mount_path in self.mount_info:
                del self.mount_info[mount_path]
                del self.memory_storage[mount_path]
            return True
        except Exception as e:
            logger.error(f"卸载内存存储失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取内存存储信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        if mount_path in self.memory_storage:
            storage_info = self.memory_storage[mount_path]
            mount_info.update({
                'used_size': storage_info['used_size'],
                'max_size': storage_info['max_size'],
                'available_size': storage_info['max_size'] - storage_info['used_size'],
                'file_count': len(storage_info['data'])
            })
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查内存存储健康状态
        """
        try:
            for mount_path in self.mount_info:
                if mount_path not in self.memory_storage:
                    return False
            return True
        except Exception:
            return False
    
    def write_file(self, mount_path: str, file_path: str, content: bytes) -> bool:
        """
        写入文件到内存存储
        """
        try:
            if mount_path not in self.memory_storage:
                return False
            
            storage = self.memory_storage[mount_path]
            file_size = len(content)
            
            # 检查空间是否足够
            if storage['used_size'] + file_size > storage['max_size']:
                logger.error(f"内存存储空间不足: {mount_path}")
                return False
            
            # 写入文件
            storage['data'][file_path] = content
            storage['used_size'] += file_size
            
            return True
            
        except Exception as e:
            logger.error(f"写入内存文件失败: {e}")
            return False
    
    def read_file(self, mount_path: str, file_path: str) -> Optional[bytes]:
        """
        从内存存储读取文件
        """
        try:
            if mount_path not in self.memory_storage:
                return None
            
            storage = self.memory_storage[mount_path]
            return storage['data'].get(file_path)
            
        except Exception as e:
            logger.error(f"读取内存文件失败: {e}")
            return None
    
    def _parse_size(self, size_str: str) -> int:
        """
        解析大小字符串
        """
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        elif size_str.endswith('TB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024 * 1024
        elif size_str.endswith('B'):
            return int(size_str[:-1])
        else:
            # 尝试直接解析为整数（字节）
            return int(size_str)


class NFSProvider(StorageProvider):
    """
    NFS存储提供者
    """
    
    def __init__(self):
        self.mount_info = {}
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载NFS存储
        """
        try:
            nfs_server = mount_config['nfs_server']
            nfs_path = mount_config['nfs_path']
            mount_path = mount_config['mount_path']
            mount_options = mount_config.get('mount_options', 'rw,sync')
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 执行mount命令
            mount_cmd = [
                'mount', '-t', 'nfs',
                '-o', mount_options,
                f"{nfs_server}:{nfs_path}",
                mount_path
            ]
            
            result = subprocess.run(mount_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"NFS挂载失败: {result.stderr}")
            
            mount_info = {
                'nfs_server': nfs_server,
                'nfs_path': nfs_path,
                'mount_path': mount_path,
                'mount_options': mount_options,
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'nfs_server': nfs_server
            }
            
        except Exception as e:
            logger.error(f"挂载NFS失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载NFS存储
        """
        try:
            if mount_path in self.mount_info:
                # 执行umount命令
                umount_cmd = ['umount', mount_path]
                result = subprocess.run(umount_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    del self.mount_info[mount_path]
                    return True
                else:
                    logger.error(f"NFS卸载失败: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"卸载NFS失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取NFS存储信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 添加文件系统信息
        if os.path.exists(mount_path):
            try:
                statvfs = os.statvfs(mount_path)
                mount_info['filesystem'] = {
                    'total_space': statvfs.f_blocks * statvfs.f_frsize,
                    'free_space': statvfs.f_bavail * statvfs.f_frsize,
                    'used_space': (statvfs.f_blocks - statvfs.f_bavail) * statvfs.f_frsize,
                    'block_size': statvfs.f_frsize
                }
            except Exception as e:
                logger.error(f"获取NFS文件系统信息失败: {e}")
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查NFS健康状态
        """
        try:
            for mount_path, mount_info in self.mount_info.items():
                # 检查挂载点是否存在
                if not os.path.exists(mount_path):
                    return False
                
                # 检查是否可以访问
                try:
                    os.listdir(mount_path)
                except Exception:
                    return False
            
            return True
        except Exception:
            return False


class S3Provider(StorageProvider):
    """
    S3存储提供者
    """
    
    def __init__(self):
        self.mount_info = {}
        self.s3_clients = {}
    
    def mount(self, mount_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        挂载S3存储
        """
        try:
            import boto3
            
            bucket_name = mount_config['bucket_name']
            mount_path = mount_config['mount_path']
            
            # 获取可选参数，使用默认值
            endpoint_url = mount_config.get('endpoint_url', 'https://s3.amazonaws.com')
            access_key = mount_config.get('access_key')
            secret_key = mount_config.get('secret_key')
            
            # 如果没有提供凭证，尝试使用环境变量或默认配置
            if not access_key:
                access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            if not secret_key:
                secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            
            # 如果仍然没有凭证，使用默认配置
            if not access_key or not secret_key:
                s3_client = boto3.client('s3')
            else:
                s3_client = boto3.client(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key
                )
            
            # 创建挂载点
            os.makedirs(mount_path, exist_ok=True)
            
            # 存储S3客户端
            self.s3_clients[mount_path] = {
                'client': s3_client,
                'bucket': bucket_name,
                'endpoint': endpoint_url
            }
            
            mount_info = {
                'endpoint_url': endpoint_url,
                'bucket_name': bucket_name,
                'mount_path': mount_path,
                'status': 'mounted'
            }
            
            self.mount_info[mount_path] = mount_info
            
            return {
                'success': True,
                'mount_path': mount_path,
                'bucket_name': bucket_name
            }
            
        except ImportError:
            logger.error("boto3未安装，无法使用S3存储")
            return {'success': False, 'error': 'boto3 not installed'}
        except Exception as e:
            logger.error(f"挂载S3失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def unmount(self, mount_path: str) -> bool:
        """
        卸载S3存储
        """
        try:
            if mount_path in self.mount_info:
                del self.mount_info[mount_path]
                if mount_path in self.s3_clients:
                    del self.s3_clients[mount_path]
            return True
        except Exception as e:
            logger.error(f"卸载S3失败: {e}")
            return False
    
    def get_info(self, mount_path: str) -> Dict[str, Any]:
        """
        获取S3存储信息
        """
        if mount_path not in self.mount_info:
            return {}
        
        mount_info = self.mount_info[mount_path]
        
        # 获取S3存储信息
        try:
            s3_info = self.s3_clients[mount_path]
            s3_client = s3_info['client']
            bucket_name = s3_info['bucket']
            
            # 获取存储桶信息
            response = s3_client.head_bucket(Bucket=bucket_name)
            
            # 获取对象数量
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            object_count = response.get('KeyCount', 0)
            
            mount_info['bucket_info'] = {
                'object_count': object_count,
                'region': response.get('LocationConstraint', 'us-east-1')
            }
            
        except Exception as e:
            logger.error(f"获取S3信息失败: {e}")
        
        return mount_info
    
    def check_health(self) -> bool:
        """
        检查S3健康状态
        """
        try:
            for mount_path, s3_info in self.s3_clients.items():
                s3_client = s3_info['client']
                bucket_name = s3_info['bucket']
                
                # 检查存储桶是否可访问
                s3_client.head_bucket(Bucket=bucket_name)
            
            return True
        except Exception:
            return False
    
    def upload_file(self, mount_path: str, local_path: str, s3_key: str) -> bool:
        """
        上传文件到S3
        """
        try:
            if mount_path not in self.s3_clients:
                return False
            
            s3_info = self.s3_clients[mount_path]
            s3_client = s3_info['client']
            bucket_name = s3_info['bucket']
            
            s3_client.upload_file(local_path, bucket_name, s3_key)
            return True
            
        except Exception as e:
            logger.error(f"上传文件到S3失败: {e}")
            return False
    
    def download_file(self, mount_path: str, s3_key: str, local_path: str) -> bool:
        """
        从S3下载文件
        """
        try:
            if mount_path not in self.s3_clients:
                return False
            
            s3_info = self.s3_clients[mount_path]
            s3_client = s3_info['client']
            bucket_name = s3_info['bucket']
            
            s3_client.download_file(bucket_name, s3_key, local_path)
            return True
            
        except Exception as e:
            logger.error(f"从S3下载文件失败: {e}")
            return False


# 存储管理器工厂类
class StorageManagerFactory:
    """
    存储管理器工厂类
    """
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> StorageProvider:
        """
        创建存储提供者
        
        Args:
            provider_type: 提供者类型
            **kwargs: 初始化参数
        
        Returns:
            存储提供者实例
        """
        providers = {
            'pvc': PVCProvider,
            'hostpath': HostPathProvider,
            'configmap': ConfigMapProvider,
            'secret': SecretProvider,
            'memory': MemoryProvider,
            'nfs': NFSProvider,
            's3': S3Provider
        }
        
        if provider_type not in providers:
            raise ValueError(f"不支持的存储提供者类型: {provider_type}")
        
        provider_class = providers[provider_type]
        return provider_class(**kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建PVC提供者
    pvc_provider = StorageManagerFactory.create_provider('pvc')
    
    # 挂载PVC
    mount_config = {
        'pvc_name': 'training-data-pvc',
        'namespace': 'default',
        'mount_path': '/mnt/training-data',
        'storage_class': 'standard',
        'access_modes': ['ReadWriteOnce'],
        'storage_size': '10Gi',
        'create_if_not_exists': True
    }
    
    result = pvc_provider.mount(mount_config)
    print(f"PVC挂载结果: {result}")
    
    # 获取存储信息
    info = pvc_provider.get_info('/mnt/training-data')
    print(f"存储信息: {info}")
    
    # 检查健康状态
    health = pvc_provider.check_health()
    print(f"健康状态: {health}") 