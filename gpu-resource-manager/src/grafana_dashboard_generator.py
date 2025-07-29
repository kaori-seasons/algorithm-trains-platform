import json
from typing import Dict, List, Any

class GrafanaDashboardGenerator:
    """Grafana仪表板生成器"""
    
    def __init__(self):
        self.dashboard_template = {
            "dashboard": {
                "id": None,
                "title": "GPU Resource Manager",
                "tags": ["gpu", "kubernetes", "resource-management"],
                "timezone": "browser",
                "panels": [],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s",
                "schemaVersion": 38,
                "version": 1
            }
        }
    
    def generate_gpu_overview_dashboard(self) -> Dict[str, Any]:
        """生成GPU概览仪表板"""
        dashboard = self.dashboard_template.copy()
        dashboard["dashboard"]["title"] = "GPU Resource Overview"
        dashboard["dashboard"]["panels"] = [
            self._create_gpu_utilization_panel(),
            self._create_gpu_memory_panel(),
            self._create_cluster_gpu_allocation_panel(),
            self._create_hpa_scaling_events_panel(),
            self._create_scheduler_queue_panel()
        ]
        return dashboard
    
    def _create_gpu_utilization_panel(self) -> Dict[str, Any]:
        """创建GPU利用率面板"""
        return {
            "id": 1,
            "title": "GPU Utilization by Node",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "gpu_utilization_percent",
                    "legendFormat": "{{node}} GPU {{gpu_id}}",
                    "refId": "A"
                }
            ],
            "yAxes": [
                {
                    "label": "Utilization %",
                    "min": 0,
                    "max": 100
                }
            ],
            "legend": {
                "show": True,
                "alignAsTable": True,
                "rightSide": True
            }
        }
    
    def _create_gpu_memory_panel(self) -> Dict[str, Any]:
        """创建GPU显存使用面板"""
        return {
            "id": 2,
            "title": "GPU Memory Usage",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [
                {
                    "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes * 100",
                    "legendFormat": "{{node}} GPU {{gpu_id}}",
                    "refId": "A"
                }
            ],
            "yAxes": [
                {
                    "label": "Memory Usage %",
                    "min": 0,
                    "max": 100
                }
            ]
        }
    
    def _create_cluster_gpu_allocation_panel(self) -> Dict[str, Any]:
        """创建集群GPU分配面板"""
        return {
            "id": 3,
            "title": "Cluster GPU Allocation",
            "type": "stat",
            "gridPos": {"h": 4, "w": 8, "x": 0, "y": 8},
            "targets": [
                {
                    "expr": "cluster_gpu_allocated / cluster_gpu_total * 100",
                    "legendFormat": "{{gpu_type}} Allocation %",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 70},
                            {"color": "red", "value": 90}
                        ]
                    }
                }
            }
        }
    
    def _create_hpa_scaling_events_panel(self) -> Dict[str, Any]:
        """创建HPA伸缩事件面板"""
        return {
            "id": 4,
            "title": "HPA Scaling Events",
            "type": "graph",
            "gridPos": {"h": 4, "w": 8, "x": 8, "y": 8},
            "targets": [
                {
                    "expr": "rate(hpa_scaling_events_total[5m])",
                    "legendFormat": "{{namespace}}/{{hpa_name}} {{direction}}",
                    "refId": "A"
                }
            ],
            "yAxes": [
                {
                    "label": "Events/sec",
                    "min": 0
                }
            ]
        }
    
    def _create_scheduler_queue_panel(self) -> Dict[str, Any]:
        """创建调度器队列面板"""
        return {
            "id": 5,
            "title": "Scheduler Queue Size",
            "type": "stat",
            "gridPos": {"h": 4, "w": 8, "x": 16, "y": 8},
            "targets": [
                {
                    "expr": "gpu_scheduler_queue_size",
                    "legendFormat": "{{priority}} Priority",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 50},
                            {"color": "red", "value": 100}
                        ]
                    }
                }
            }
        }
    
    def generate_gpu_resource_dashboard(self) -> Dict[str, Any]:
        """生成GPU资源详细仪表板"""
        dashboard = self.dashboard_template.copy()
        dashboard["dashboard"]["title"] = "GPU Resource Details"
        dashboard["dashboard"]["panels"] = [
            self._create_gpu_temperature_panel(),
            self._create_gpu_power_panel(),
            self._create_gpu_memory_breakdown_panel(),
            self._create_node_gpu_allocation_panel(),
            self._create_scheduling_latency_panel()
        ]
        return dashboard
    
    def _create_gpu_temperature_panel(self) -> Dict[str, Any]:
        """创建GPU温度面板"""
        return {
            "id": 6,
            "title": "GPU Temperature",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            "targets": [
                {
                    "expr": "gpu_temperature_celsius",
                    "legendFormat": "{{node}} GPU {{gpu_id}}",
                    "refId": "A"
                }
            ],
            "yAxes": [
                {
                    "label": "Temperature (°C)",
                    "min": 0,
                    "max": 100
                }
            ],
            "thresholds": [
                {"value": 80, "colorMode": "critical", "op": "gt"}
            ]
        }
    
    def _create_gpu_power_panel(self) -> Dict[str, Any]:
        """创建GPU功耗面板"""
        return {
            "id": 7,
            "title": "GPU Power Usage",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            "targets": [
                {
                    "expr": "gpu_power_usage_watts",
                    "legendFormat": "{{node}} GPU {{gpu_id}}",
                    "refId": "A"
                }
            ],
            "yAxes": [
                {
                    "label": "Power (W)",
                    "min": 0
                }
            ]
        }
    
    def _create_gpu_memory_breakdown_panel(self) -> Dict[str, Any]:
        """创建GPU显存分解面板"""
        return {
            "id": 8,
            "title": "GPU Memory Breakdown",
            "type": "piechart",
            "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
            "targets": [
                {
                    "expr": "gpu_memory_used_bytes",
                    "legendFormat": "Used - {{node}} GPU {{gpu_id}}",
                    "refId": "A"
                },
                {
                    "expr": "gpu_memory_total_bytes - gpu_memory_used_bytes",
                    "legendFormat": "Free - {{node}} GPU {{gpu_id}}",
                    "refId": "B"
                }
            ]
        }
    
    def _create_node_gpu_allocation_panel(self) -> Dict[str, Any]:
        """创建节点GPU分配面板"""
        return {
            "id": 9,
            "title": "Node GPU Allocation",
            "type": "table",
            "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
            "targets": [
                {
                    "expr": "cluster_gpu_total",
                    "format": "table",
                    "refId": "A"
                },
                {
                    "expr": "cluster_gpu_allocated",
                    "format": "table",
                    "refId": "B"
                }
            ],
            "transformations": [
                {
                    "id": "merge",
                    "options": {}
                }
            ]
        }
    
    def _create_scheduling_latency_panel(self) -> Dict[str, Any]:
        """创建调度延迟面板"""
        return {
            "id": 10,
            "title": "GPU Scheduling Latency",
            "type": "graph",
            "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
            "targets": [
                {
                    "expr": "histogram_quantile(0.95, gpu_scheduling_latency_seconds_bucket)",
                    "legendFormat": "95th percentile",
                    "refId": "A"
                },
                {
                    "expr": "histogram_quantile(0.50, gpu_scheduling_latency_seconds_bucket)",
                    "legendFormat": "50th percentile",
                    "refId": "B"
                }
            ],
            "yAxes": [
                {
                    "label": "Latency (s)",
                    "min": 0
                }
            ]
        }
    
    def save_dashboard_to_file(self, dashboard: Dict[str, Any], filename: str):
        """保存仪表板到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False) 