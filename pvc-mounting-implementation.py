#!/usr/bin/env python3
"""
PVC挂载实现示例
展示在Kubernetes环境中正确挂载PVC的方法
"""

import os
import subprocess
import time
import json
from typing import Dict, Any, Optional, List
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PVCManager:
    """
    PVC管理器
    提供PVC的创建、挂载和管理功能
    """
    
    def __init__(self, kube_config_path: Optional[str] = None):
        """
        初始化PVC管理器
        
        Args:
            kube_config_path: Kubernetes配置文件路径
        """
        self.kube_config_path = kube_config_path
        self.core_v1_api = None
        self.apps_v1_api = None
        self._init_kubernetes_client()
        
        # 存储挂载信息
        self.mount_info = {}
    
    def _init_kubernetes_client(self):
        """初始化Kubernetes客户端"""
        try:
            if self.kube_config_path and os.path.exists(self.kube_config_path):
                config.load_kube_config(config_file=self.kube_config_path)
            else:
                config.load_incluster_config()
            
            self.core_v1_api = client.CoreV1Api()
            self.apps_v1_api = client.AppsV1Api()
            logger.info("Kubernetes客户端初始化成功")
            
        except Exception as e:
            logger.error(f"Kubernetes客户端初始化失败: {e}")
            raise
    
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
                    'labels': {'app': 'train-platform'}
                }
        
        Returns:
            PVC名称
        """
        try:
            pvc = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(
                    name=pvc_config['name'],
                    namespace=pvc_config.get('namespace', 'default'),
                    labels=pvc_config.get('labels', {})
                ),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=pvc_config['access_modes'],
                    resources=client.V1ResourceRequirements(
                        requests={'storage': pvc_config['storage_size']}
                    ),
                    storage_class_name=pvc_config.get('storage_class')
                )
            )
            
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
    
    def mount_pvc_in_pod(self, pvc_name: str, namespace: str, mount_path: str, 
                        pod_name: str, container_image: str = "busybox:latest") -> Dict[str, Any]:
        """
        在Pod中挂载PVC（推荐方式）
        
        Args:
            pvc_name: PVC名称
            namespace: 命名空间
            mount_path: 容器内的挂载路径
            pod_name: Pod名称
            container_image: 容器镜像
        
        Returns:
            挂载结果
        """
        try:
            # 创建Pod配置
            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=pod_name,
                    namespace=namespace
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name="main-container",
                            image=container_image,
                            command=["sleep", "3600"],  # 保持Pod运行
                            volume_mounts=[
                                client.V1VolumeMount(
                                    name="pvc-volume",
                                    mount_path=mount_path
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
            
            # 创建Pod
            created_pod = self.core_v1_api.create_namespaced_pod(
                namespace=namespace,
                body=pod
            )
            
            # 等待Pod运行
            self._wait_for_pod_ready(pod_name, namespace)
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': pvc_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'pod_name': pod_name,
                'mount_type': 'pod_mount',
                'status': 'mounted'
            }
            
            self.mount_info[f"{namespace}/{pod_name}"] = mount_info
            
            logger.info(f"PVC {pvc_name} 在Pod {pod_name} 中挂载成功")
            
            return {
                'success': True,
                'pod_name': pod_name,
                'namespace': namespace,
                'mount_path': mount_path,
                'mount_info': mount_info
            }
            
        except Exception as e:
            logger.error(f"Pod挂载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def mount_pvc_on_host(self, pvc_name: str, namespace: str, host_mount_path: str) -> Dict[str, Any]:
        """
        在宿主机上挂载PVC（通过临时Pod和kubectl cp）
        
        Args:
            pvc_name: PVC名称
            namespace: 命名空间
            host_mount_path: 宿主机挂载路径
        
        Returns:
            挂载结果
        """
        try:
            # 创建临时Pod来访问PVC
            temp_pod_name = f"temp-mount-{pvc_name}-{int(time.time())}"
            
            # 在Pod中挂载PVC
            pod_mount_result = self.mount_pvc_in_pod(
                pvc_name=pvc_name,
                namespace=namespace,
                mount_path="/data",  # Pod内的挂载路径
                pod_name=temp_pod_name,
                container_image="busybox:latest"
            )
            
            if not pod_mount_result['success']:
                raise Exception(f"临时Pod挂载失败: {pod_mount_result['error']}")
            
            # 创建宿主机挂载点
            os.makedirs(host_mount_path, exist_ok=True)
            
            # 使用kubectl cp复制数据到宿主机
            kubectl_cp_cmd = [
                'kubectl', 'cp',
                f"{namespace}/{temp_pod_name}:/data",
                host_mount_path
            ]
            
            result = subprocess.run(kubectl_cp_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"kubectl cp失败: {result.stderr}")
            
            # 删除临时Pod
            self._delete_pod(temp_pod_name, namespace)
            
            # 记录挂载信息
            mount_info = {
                'pvc_name': pvc_name,
                'namespace': namespace,
                'host_mount_path': host_mount_path,
                'temp_pod_name': temp_pod_name,
                'mount_type': 'host_mount',
                'status': 'mounted'
            }
            
            self.mount_info[host_mount_path] = mount_info
            
            logger.info(f"PVC {pvc_name} 在宿主机 {host_mount_path} 挂载成功")
            
            return {
                'success': True,
                'host_mount_path': host_mount_path,
                'pvc_name': pvc_name,
                'mount_info': mount_info
            }
            
        except Exception as e:
            logger.error(f"宿主机挂载失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_deployment_with_pvc(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建带有PVC挂载的Deployment
        
        Args:
            deployment_config: Deployment配置
                {
                    'name': 'training-deployment',
                    'namespace': 'default',
                    'replicas': 1,
                    'image': 'train-platform/training:latest',
                    'pvc_mounts': [
                        {
                            'pvc_name': 'training-data-pvc',
                            'mount_path': '/mnt/training-data'
                        }
                    ]
                }
        
        Returns:
            创建结果
        """
        try:
            # 构建volume mounts
            volume_mounts = []
            volumes = []
            
            for i, pvc_mount in enumerate(deployment_config['pvc_mounts']):
                volume_name = f"pvc-volume-{i}"
                
                volume_mounts.append(
                    client.V1VolumeMount(
                        name=volume_name,
                        mount_path=pvc_mount['mount_path']
                    )
                )
                
                volumes.append(
                    client.V1Volume(
                        name=volume_name,
                        persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=pvc_mount['pvc_name']
                        )
                    )
                )
            
            # 创建Deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=deployment_config['name'],
                    namespace=deployment_config.get('namespace', 'default')
                ),
                spec=client.V1DeploymentSpec(
                    replicas=deployment_config.get('replicas', 1),
                    selector=client.V1LabelSelector(
                        match_labels={'app': deployment_config['name']}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={'app': deployment_config['name']}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="main-container",
                                    image=deployment_config['image'],
                                    volume_mounts=volume_mounts
                                )
                            ],
                            volumes=volumes
                        )
                    )
                )
            )
            
            namespace = deployment_config.get('namespace', 'default')
            created_deployment = self.apps_v1_api.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Deployment {deployment_config['name']} 创建成功")
            
            return {
                'success': True,
                'deployment_name': deployment_config['name'],
                'namespace': namespace
            }
            
        except Exception as e:
            logger.error(f"创建Deployment失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _wait_for_pod_ready(self, pod_name: str, namespace: str, timeout: int = 300) -> bool:
        """等待Pod就绪"""
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
    
    def _delete_pod(self, pod_name: str, namespace: str):
        """删除Pod"""
        try:
            self.core_v1_api.delete_namespaced_pod(pod_name, namespace)
        except Exception as e:
            logger.warning(f"删除Pod失败: {e}")
    
    def unmount_pvc(self, mount_identifier: str) -> bool:
        """
        卸载PVC
        
        Args:
            mount_identifier: 挂载标识符（Pod名称或宿主机路径）
        
        Returns:
            是否卸载成功
        """
        try:
            if mount_identifier not in self.mount_info:
                logger.warning(f"挂载标识符 {mount_identifier} 未找到")
                return True
            
            mount_info = self.mount_info[mount_identifier]
            mount_type = mount_info.get('mount_type')
            
            if mount_type == 'pod_mount':
                # 删除Pod
                pod_name = mount_info['pod_name']
                namespace = mount_info['namespace']
                self._delete_pod(pod_name, namespace)
                
            elif mount_type == 'host_mount':
                # 清理宿主机挂载点
                host_mount_path = mount_info['host_mount_path']
                if os.path.exists(host_mount_path):
                    import shutil
                    shutil.rmtree(host_mount_path, ignore_errors=True)
            
            # 从记录中删除
            del self.mount_info[mount_identifier]
            
            logger.info(f"PVC卸载成功: {mount_identifier}")
            return True
            
        except Exception as e:
            logger.error(f"卸载PVC失败: {e}")
            return False
    
    def list_mounts(self) -> List[Dict[str, Any]]:
        """列出所有挂载"""
        return list(self.mount_info.values())
    
    def get_pvc_status(self, pvc_name: str, namespace: str = 'default') -> Dict[str, Any]:
        """获取PVC状态"""
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
                'volume_name': pvc.spec.volume_name
            }
            
        except ApiException as e:
            if e.status == 404:
                return {}
            else:
                raise

# 使用示例
def main():
    """使用示例"""
    
    # 创建PVC管理器
    pvc_manager = PVCManager()
    
    # 1. 创建PVC
    pvc_config = {
        'name': 'training-data-pvc',
        'namespace': 'default',
        'storage_class': 'standard',
        'access_modes': ['ReadWriteOnce'],
        'storage_size': '10Gi',
        'labels': {'app': 'train-platform'}
    }
    
    pvc_name = pvc_manager.create_pvc(pvc_config)
    print(f"PVC创建成功: {pvc_name}")
    
    # 2. 在Pod中挂载PVC
    pod_mount_result = pvc_manager.mount_pvc_in_pod(
        pvc_name='training-data-pvc',
        namespace='default',
        mount_path='/mnt/training-data',
        pod_name='training-pod'
    )
    
    if pod_mount_result['success']:
        print(f"Pod挂载成功: {pod_mount_result}")
    else:
        print(f"Pod挂载失败: {pod_mount_result['error']}")
    
    # 3. 在宿主机上挂载PVC
    host_mount_result = pvc_manager.mount_pvc_on_host(
        pvc_name='training-data-pvc',
        namespace='default',
        host_mount_path='/mnt/host-training-data'
    )
    
    if host_mount_result['success']:
        print(f"宿主机挂载成功: {host_mount_result}")
    else:
        print(f"宿主机挂载失败: {host_mount_result['error']}")
    
    # 4. 创建带有PVC挂载的Deployment
    deployment_config = {
        'name': 'training-deployment',
        'namespace': 'default',
        'replicas': 1,
        'image': 'train-platform/training:latest',
        'pvc_mounts': [
            {
                'pvc_name': 'training-data-pvc',
                'mount_path': '/mnt/training-data'
            }
        ]
    }
    
    deployment_result = pvc_manager.create_deployment_with_pvc(deployment_config)
    
    if deployment_result['success']:
        print(f"Deployment创建成功: {deployment_result}")
    else:
        print(f"Deployment创建失败: {deployment_result['error']}")
    
    # 5. 列出所有挂载
    mounts = pvc_manager.list_mounts()
    print(f"当前挂载: {json.dumps(mounts, indent=2)}")

if __name__ == "__main__":
    main() 