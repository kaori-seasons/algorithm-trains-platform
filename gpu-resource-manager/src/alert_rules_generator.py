import yaml
from typing import Dict, List, Any

class AlertRulesGenerator:
    """告警规则生成器"""
    
    def __init__(self):
        self.rule_template = {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PrometheusRule",
            "metadata": {
                "name": "gpu-resource-manager-alerts",
                "namespace": "kube-system",
                "labels": {
                    "app": "gpu-resource-manager",
                    "prometheus": "kube-prometheus"
                }
            },
            "spec": {
                "groups": []
            }
        }
    
    def generate_gpu_alert_rules(self) -> Dict[str, Any]:
        """生成GPU相关告警规则"""
        rules = self.rule_template.copy()
        rules["spec"]["groups"] = [
            self._create_gpu_utilization_rules(),
            self._create_gpu_memory_rules(),
            self._create_gpu_temperature_rules(),
            self._create_scheduler_rules(),
            self._create_hpa_rules()
        ]
        return rules
    
    def _create_gpu_utilization_rules(self) -> Dict[str, Any]:
        """创建GPU利用率告警规则"""
        return {
            "name": "gpu-utilization",
            "interval": "30s",
            "rules": [
                {
                    "alert": "GPUHighUtilization",
                    "expr": "gpu_utilization_percent > 90",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "GPU利用率过高",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 利用率已超过90%，当前值: {{ $value }}%"
                    }
                },
                {
                    "alert": "GPULowUtilization",
                    "expr": "gpu_utilization_percent < 10",
                    "for": "30m",
                    "labels": {
                        "severity": "info"
                    },
                    "annotations": {
                        "summary": "GPU利用率过低",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 利用率低于10%超过30分钟，当前值: {{ $value }}%"
                    }
                }
            ]
        }
    
    def _create_gpu_memory_rules(self) -> Dict[str, Any]:
        """创建GPU显存告警规则"""
        return {
            "name": "gpu-memory",
            "interval": "30s",
            "rules": [
                {
                    "alert": "GPUMemoryHigh",
                    "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "GPU显存使用率过高",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 显存使用率超过90%"
                    }
                },
                {
                    "alert": "GPUMemoryFull",
                    "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95",
                    "for": "2m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "GPU显存几乎耗尽",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 显存使用率超过95%，可能导致OOM"
                    }
                }
            ]
        }
    
    def _create_gpu_temperature_rules(self) -> Dict[str, Any]:
        """创建GPU温度告警规则"""
        return {
            "name": "gpu-temperature",
            "interval": "30s",
            "rules": [
                {
                    "alert": "GPUHighTemperature",
                    "expr": "gpu_temperature_celsius > 80",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "GPU温度过高",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 温度超过80°C，当前温度: {{ $value }}°C"
                    }
                },
                {
                    "alert": "GPUCriticalTemperature",
                    "expr": "gpu_temperature_celsius > 90",
                    "for": "1m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "GPU温度危险",
                        "description": "节点 {{ $labels.node }} 的 GPU {{ $labels.gpu_id }} 温度超过90°C，可能导致硬件损坏"
                    }
                }
            ]
        }
    
    def _create_scheduler_rules(self) -> Dict[str, Any]:
        """创建调度器告警规则"""
        return {
            "name": "gpu-scheduler",
            "interval": "30s",
            "rules": [
                {
                    "alert": "SchedulerQueueHigh",
                    "expr": "gpu_scheduler_queue_size > 100",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "调度队列积压过多",
                        "description": "GPU调度器队列中有 {{ $value }} 个待处理任务"
                    }
                },
                {
                    "alert": "SchedulingLatencyHigh",
                    "expr": "histogram_quantile(0.95, gpu_scheduling_latency_seconds_bucket) > 10",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "调度延迟过高",
                        "description": "GPU调度延迟95分位数超过10秒"
                    }
                }
            ]
        }
    
    def _create_hpa_rules(self) -> Dict[str, Any]:
        """创建HPA告警规则"""
        return {
            "name": "gpu-hpa",
            "interval": "30s",
            "rules": [
                {
                    "alert": "HPAScalingFrequent",
                    "expr": "rate(hpa_scaling_events_total[10m]) > 0.1",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "HPA频繁伸缩",
                        "description": "HPA {{ $labels.namespace }}/{{ $labels.hpa_name }} 在10分钟内伸缩频率超过0.1次/分钟"
                    }
                }
            ]
        }
    
    def save_rules_to_file(self, rules: Dict[str, Any], filename: str):
        """保存告警规则到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(rules, f, default_flow_style=False, allow_unicode=True) 