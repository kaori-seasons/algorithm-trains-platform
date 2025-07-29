import time
import json
from stage3_main import Stage3Main

def example_stage3_usage():
    """第三阶段使用示例"""
    # 初始化第三阶段管理器
    stage3_main = Stage3Main()
    
    try:
        # 启动服务（在后台线程中）
        import threading
        service_thread = threading.Thread(target=stage3_main.start, daemon=True)
        service_thread.start()
        
        # 等待服务启动
        time.sleep(5)
        
        # 示例1: 获取监控状态
        print("获取监控状态...")
        status = stage3_main.get_status()
        print(f"Stage2状态: {status['stage2_status']}")
        print(f"Stage3状态: {status['stage3_status']}")
        
        # 示例2: 获取集群摘要
        print("获取集群摘要...")
        summary = stage3_main.stage3_manager.get_cluster_summary()
        if summary:
            print(f"总GPU数量: {summary['current']['total_gpus']}")
            print(f"已分配GPU: {summary['current']['allocated_gpus']}")
            print(f"平均利用率: {summary['current']['avg_utilization']}%")
            print(f"平均显存使用: {summary['current']['avg_memory_usage']}%")
        
        # 示例3: 获取资源趋势
        print("获取资源趋势...")
        trends = stage3_main.stage3_manager.get_resource_trends()
        if trends:
            print(f"趋势数据点数量: {len(trends['timestamps'])}")
            print(f"GPU利用率趋势: {trends['gpu_utilization'][:5]}...")  # 显示前5个数据点
        
        # 示例4: 获取节点性能排名
        print("获取节点性能排名...")
        performance = stage3_main.stage3_manager.get_node_performance()
        if performance['nodes']:
            print("节点性能排名:")
            for i, node in enumerate(performance['nodes'][:3]):  # 显示前3名
                print(f"  {i+1}. {node['node_name']}: 分数={node['performance_score']}, "
                      f"利用率={node['utilization']}%, 显存={node['memory_usage']}%")
        
        # 示例5: 获取告警摘要
        print("获取告警摘要...")
        alert_summary = stage3_main.stage3_manager.get_alert_summary()
        print(f"总告警数: {alert_summary['total']}")
        print(f"严重告警: {alert_summary['critical']}")
        print(f"警告告警: {alert_summary['warning']}")
        print(f"信息告警: {alert_summary['info']}")
        
        # 示例6: 导出仪表板配置
        print("导出仪表板配置...")
        overview_config = stage3_main.stage3_manager.export_dashboard_config("overview")
        detail_config = stage3_main.stage3_manager.export_dashboard_config("detail")
        
        print(f"概览仪表板面板数: {len(overview_config['dashboard']['panels'])}")
        print(f"详细仪表板面板数: {len(detail_config['dashboard']['panels'])}")
        
        # 保存仪表板配置到文件
        with open('gpu_overview_dashboard.json', 'w', encoding='utf-8') as f:
            json.dump(overview_config, f, indent=2, ensure_ascii=False)
        
        with open('gpu_detail_dashboard.json', 'w', encoding='utf-8') as f:
            json.dump(detail_config, f, indent=2, ensure_ascii=False)
        
        print("仪表板配置已保存到文件")
        
        # 示例7: 生成监控报告
        print("生成监控报告...")
        report = stage3_main.stage3_manager.generate_monitoring_report()
        if report:
            print(f"报告时间戳: {report['timestamp']}")
            print(f"总GPU数: {report['summary']['total_gpus']}")
            print(f"已分配GPU: {report['summary']['allocated_gpus']}")
            print(f"平均利用率: {report['summary']['avg_utilization']}%")
            print(f"活跃Pod数: {report['summary']['active_pods']}")
            print(f"待处理Pod数: {report['summary']['pending_pods']}")
            print(f"总告警数: {report['summary']['total_alerts']}")
        
        # 示例8: 记录自定义指标
        print("记录自定义指标...")
        stage3_main.stage3_manager.record_custom_metric(
            "custom_gpu_utilization",
            85.5,
            {"node": "gpu-node-1", "gpu_id": "0", "gpu_type": "V100"}
        )
        print("自定义指标已记录")
        
        # 示例9: 获取性能指标
        print("获取性能指标...")
        metrics = stage3_main.stage3_manager.get_performance_metrics()
        print(f"性能指标包含 {len(metrics)} 个主要组件")
        
        # 示例10: 模拟Web API访问
        print("模拟Web API访问...")
        try:
            import requests
            
            # 健康检查
            response = requests.get('http://localhost:8080/health', timeout=5)
            print(f"健康检查状态: {response.status_code}")
            
            # 获取GPU指标
            response = requests.get('http://localhost:8080/api/v1/gpu/metrics', timeout=5)
            print(f"GPU指标API状态: {response.status_code}")
            
            # 获取集群状态
            response = requests.get('http://localhost:8080/api/v1/cluster/status', timeout=5)
            print(f"集群状态API状态: {response.status_code}")
            
        except requests.exceptions.RequestException as e:
            print(f"Web API访问失败: {e}")
        
        # 等待一段时间观察系统运行
        print("系统运行中，等待30秒...")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("收到中断信号")
    finally:
        # 停止服务
        stage3_main.stop()

def example_monitoring_dashboard():
    """监控仪表板示例"""
    from src.grafana_dashboard_generator import GrafanaDashboardGenerator
    
    print("生成Grafana仪表板配置...")
    generator = GrafanaDashboardGenerator()
    
    # 生成概览仪表板
    overview_dashboard = generator.generate_gpu_overview_dashboard()
    print(f"概览仪表板标题: {overview_dashboard['dashboard']['title']}")
    print(f"概览仪表板面板数: {len(overview_dashboard['dashboard']['panels'])}")
    
    # 生成详细仪表板
    detail_dashboard = generator.generate_gpu_resource_dashboard()
    print(f"详细仪表板标题: {detail_dashboard['dashboard']['title']}")
    print(f"详细仪表板面板数: {len(detail_dashboard['dashboard']['panels'])}")
    
    # 保存到文件
    generator.save_dashboard_to_file(overview_dashboard, 'gpu_overview_dashboard.json')
    generator.save_dashboard_to_file(detail_dashboard, 'gpu_detail_dashboard.json')
    print("仪表板配置已保存到文件")

def example_alert_rules():
    """告警规则示例"""
    from src.alert_rules_generator import AlertRulesGenerator
    
    print("生成告警规则...")
    generator = AlertRulesGenerator()
    
    # 生成GPU告警规则
    alert_rules = generator.generate_gpu_alert_rules()
    print(f"告警规则组数: {len(alert_rules['spec']['groups'])}")
    
    # 保存到文件
    generator.save_rules_to_file(alert_rules, 'gpu_alert_rules.yaml')
    print("告警规则已保存到文件")

if __name__ == "__main__":
    print("=== 第三阶段使用示例 ===")
    
    # 运行主要示例
    example_stage3_usage()
    
    # 运行仪表板示例
    example_monitoring_dashboard()
    
    # 运行告警规则示例
    example_alert_rules()
    
    print("所有示例运行完成！") 