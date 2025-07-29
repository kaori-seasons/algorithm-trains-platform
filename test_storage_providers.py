#!/usr/bin/env python3
"""
存储提供者测试脚本
验证所有存储提供者的基本功能
"""

import os
import tempfile
import shutil
from storage_providers import StorageManagerFactory

def test_hostpath_provider():
    """测试HostPath提供者"""
    print("\n🔧 测试 HostPath 提供者...")
    
    provider = StorageManagerFactory.create_provider('hostpath')
    
    # 创建临时目录作为挂载点
    with tempfile.TemporaryDirectory() as temp_dir:
        mount_config = {
            'host_path': temp_dir,
            'mount_path': '/tmp/test-hostpath'
        }
        
        # 测试挂载
        result = provider.mount(mount_config)
        print(f"  挂载结果: {result['success']}")
        
        if result['success']:
            # 测试获取信息
            info = provider.get_info('/tmp/test-hostpath')
            print(f"  挂载信息: {info}")
            
            # 测试健康检查
            health = provider.check_health()
            print(f"  健康状态: {health}")
            
            # 测试卸载
            unmount_result = provider.unmount('/tmp/test-hostpath')
            print(f"  卸载结果: {unmount_result}")

def test_memory_provider():
    """测试Memory提供者"""
    print("\n🔧 测试 Memory 提供者...")
    
    provider = StorageManagerFactory.create_provider('memory')
    
    mount_config = {
        'mount_path': '/tmp/test-memory',
        'max_size': '100MB'  # 使用正确的参数名
    }
    
    # 测试挂载
    result = provider.mount(mount_config)
    print(f"  挂载结果: {result['success']}")
    
    if result['success']:
        # 测试写入文件
        write_result = provider.write_file('/tmp/test-memory', 'test.txt', b'Hello, Memory Storage!')
        print(f"  写入文件: {write_result}")
        
        # 测试读取文件
        content = provider.read_file('/tmp/test-memory', 'test.txt')
        print(f"  读取文件: {content.decode() if content else 'None'}")
        
        # 测试获取信息
        info = provider.get_info('/tmp/test-memory')
        print(f"  挂载信息: {info}")
        
        # 测试健康检查
        health = provider.check_health()
        print(f"  健康状态: {health}")
        
        # 测试卸载
        unmount_result = provider.unmount('/tmp/test-memory')
        print(f"  卸载结果: {unmount_result}")

def test_nfs_provider():
    """测试NFS提供者"""
    print("\n🔧 测试 NFS 提供者...")
    
    provider = StorageManagerFactory.create_provider('nfs')
    
    # 注意：这里使用一个不存在的NFS服务器，应该会失败
    mount_config = {
        'nfs_server': '192.168.1.100',
        'nfs_path': '/shared',
        'mount_path': '/tmp/test-nfs'
    }
    
    # 测试挂载（预期会失败，因为没有真实的NFS服务器）
    result = provider.mount(mount_config)
    print(f"  挂载结果: {result['success']}")
    if not result['success']:
        print(f"  错误信息: {result.get('error', 'Unknown error')}")

def test_s3_provider():
    """测试S3提供者"""
    print("\n🔧 测试 S3 提供者...")
    
    provider = StorageManagerFactory.create_provider('s3')
    
    # 注意：这里使用默认配置，需要AWS凭证
    mount_config = {
        'bucket_name': 'test-bucket',
        'mount_path': '/tmp/test-s3'
    }
    
    # 测试挂载（预期会失败，因为没有AWS凭证）
    result = provider.mount(mount_config)
    print(f"  挂载结果: {result['success']}")
    if not result['success']:
        print(f"  错误信息: {result.get('error', 'Unknown error')}")

def test_kubernetes_providers():
    """测试Kubernetes相关提供者"""
    print("\n🔧 测试 Kubernetes 提供者...")
    
    providers = ['pvc', 'configmap', 'secret']
    
    for provider_type in providers:
        print(f"\n  测试 {provider_type.upper()} 提供者...")
        try:
            provider = StorageManagerFactory.create_provider(provider_type)
            print(f"    ✅ {provider_type} 提供者创建成功")
            
            # 测试健康检查
            health = provider.check_health()
            print(f"    健康状态: {health}")
            
        except Exception as e:
            print(f"    ❌ {provider_type} 提供者创建失败: {e}")

def test_storage_manager_factory():
    """测试存储管理器工厂"""
    print("\n🔧 测试 StorageManagerFactory...")
    
    # 测试支持的提供者类型
    supported_types = ['pvc', 'hostpath', 'configmap', 'secret', 'memory', 'nfs', 's3']
    
    for provider_type in supported_types:
        try:
            provider = StorageManagerFactory.create_provider(provider_type)
            print(f"  ✅ {provider_type}: {type(provider).__name__}")
        except Exception as e:
            print(f"  ❌ {provider_type}: {e}")
    
    # 测试不支持的提供者类型
    try:
        provider = StorageManagerFactory.create_provider('unsupported')
        print(f"  ❌ 不支持的提供者类型应该失败")
    except ValueError as e:
        print(f"  ✅ 正确拒绝了不支持的提供者类型: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试存储提供者...")
    
    # 测试存储管理器工厂
    test_storage_manager_factory()
    
    # 测试各个提供者
    test_hostpath_provider()
    test_memory_provider()
    test_nfs_provider()
    test_s3_provider()
    test_kubernetes_providers()
    
    print("\n✅ 所有测试完成！")

if __name__ == "__main__":
    main() 