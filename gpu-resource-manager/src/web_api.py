from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from typing import Dict, Any, List
from datetime import datetime

class MonitoringWebAPI:
    """监控Web API"""
    
    def __init__(self, gpu_exporter, aggregator, stage2_manager):
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.gpu_exporter = gpu_exporter
        self.aggregator = aggregator
        self.stage2_manager = stage2_manager
        self.logger = logging.getLogger(__name__)
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/api/v1/gpu/metrics', methods=['GET'])
        def get_gpu_metrics():
            """获取GPU指标"""
            try:
                node_name = request.args.get('node')
                metrics = self.stage2_manager.get_gpu_metrics(node_name)
                return jsonify({
                    'status': 'success',
                    'data': metrics
                })
            except Exception as e:
                self.logger.error(f"获取GPU指标失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/cluster/status', methods=['GET'])
        def get_cluster_status():
            """获取集群状态"""
            try:
                status = self.stage2_manager.get_system_status()
                return jsonify({
                    'status': 'success',
                    'data': status
                })
            except Exception as e:
                self.logger.error(f"获取集群状态失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/trends', methods=['GET'])
        def get_resource_trends():
            """获取资源趋势"""
            try:
                hours = int(request.args.get('hours', 24))
                trends = self.aggregator.get_resource_trends()
                return jsonify({
                    'status': 'success',
                    'data': trends
                })
            except Exception as e:
                self.logger.error(f"获取资源趋势失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/hpa', methods=['POST'])
        def create_hpa():
            """创建HPA"""
            try:
                data = request.get_json()
                success = self.stage2_manager.create_gpu_aware_hpa(
                    namespace=data['namespace'],
                    deployment_name=data['deployment_name'],
                    min_replicas=data['min_replicas'],
                    max_replicas=data['max_replicas'],
                    cpu_threshold=data.get('cpu_threshold', 70),
                    memory_threshold=data.get('memory_threshold', 80),
                    gpu_threshold=data.get('gpu_threshold', 60)
                )
                
                return jsonify({
                    'status': 'success' if success else 'error',
                    'message': 'HPA创建成功' if success else 'HPA创建失败'
                })
            except Exception as e:
                self.logger.error(f"创建HPA失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/schedule', methods=['POST'])
        def submit_scheduling_request():
            """提交调度请求"""
            try:
                data = request.get_json()
                success = self.stage2_manager.submit_scheduling_request(
                    pod_name=data['pod_name'],
                    namespace=data['namespace'],
                    gpu_requirement=data['gpu_requirement'],
                    memory_requirement=data['memory_requirement'],
                    priority=data.get('priority', 1),
                    node_selector=data.get('node_selector')
                )
                
                return jsonify({
                    'status': 'success' if success else 'error',
                    'message': '调度请求提交成功' if success else '调度请求提交失败'
                })
            except Exception as e:
                self.logger.error(f"提交调度请求失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/summary', methods=['GET'])
        def get_cluster_summary():
            """获取集群摘要"""
            try:
                summary = self.aggregator.get_cluster_summary()
                return jsonify({
                    'status': 'success',
                    'data': summary
                })
            except Exception as e:
                self.logger.error(f"获取集群摘要失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/nodes/performance', methods=['GET'])
        def get_node_performance():
            """获取节点性能排名"""
            try:
                performance = self.aggregator.get_node_performance_ranking()
                return jsonify({
                    'status': 'success',
                    'data': performance
                })
            except Exception as e:
                self.logger.error(f"获取节点性能失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/alerts', methods=['GET'])
        def get_alerts():
            """获取告警信息"""
            try:
                # 模拟告警数据
                alerts = self._get_simulated_alerts()
                return jsonify({
                    'status': 'success',
                    'data': alerts
                })
            except Exception as e:
                self.logger.error(f"获取告警失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/api/v1/dashboard/config', methods=['GET'])
        def get_dashboard_config():
            """获取仪表板配置"""
            try:
                from grafana_dashboard_generator import GrafanaDashboardGenerator
                generator = GrafanaDashboardGenerator()
                
                dashboard_type = request.args.get('type', 'overview')
                if dashboard_type == 'overview':
                    config = generator.generate_gpu_overview_dashboard()
                else:
                    config = generator.generate_gpu_resource_dashboard()
                
                return jsonify({
                    'status': 'success',
                    'data': config
                })
            except Exception as e:
                self.logger.error(f"获取仪表板配置失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Prometheus指标端点"""
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from flask import Response
            
            return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    
    def _get_simulated_alerts(self) -> List[Dict[str, Any]]:
        """获取模拟告警数据"""
        import random
        
        alert_types = [
            {
                'name': 'GPUHighUtilization',
                'severity': 'warning',
                'description': 'GPU利用率过高',
                'count': random.randint(0, 3)
            },
            {
                'name': 'GPUMemoryHigh',
                'severity': 'warning',
                'description': 'GPU显存使用率过高',
                'count': random.randint(0, 2)
            },
            {
                'name': 'GPUHighTemperature',
                'severity': 'critical',
                'description': 'GPU温度过高',
                'count': random.randint(0, 1)
            },
            {
                'name': 'SchedulerQueueHigh',
                'severity': 'warning',
                'description': '调度队列积压过多',
                'count': random.randint(0, 1)
            }
        ]
        
        alerts = []
        for alert_type in alert_types:
            if alert_type['count'] > 0:
                for i in range(alert_type['count']):
                    alerts.append({
                        'id': f"{alert_type['name']}_{i}",
                        'name': alert_type['name'],
                        'severity': alert_type['severity'],
                        'description': alert_type['description'],
                        'timestamp': datetime.now().isoformat(),
                        'node': f"node-{random.randint(1, 5)}",
                        'gpu_id': str(random.randint(0, 3))
                    })
        
        return alerts
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """启动Web服务"""
        self.app.run(host=host, port=port, debug=debug) 