"""
质量评估服务
用于评估振动数据质量并提供重新预处理功能
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QualityIssueType(Enum):
    """质量问题类型"""
    DISCONTINUOUS_WAVEFORM = "discontinuous_waveform"  # 波形不连续
    NOISE_POLLUTION = "noise_pollution"  # 噪声污染
    MISSING_DATA = "missing_data"  # 数据缺失
    ABNORMAL_AMPLITUDE = "abnormal_amplitude"  # 异常振幅
    FREQUENCY_DRIFT = "frequency_drift"  # 频率漂移
    PHASE_SHIFT = "phase_shift"  # 相位偏移


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 差
    UNUSABLE = "unusable"  # 不可用


@dataclass
class QualityIssue:
    """质量问题"""
    issue_type: QualityIssueType
    severity: float  # 严重程度 0-1
    start_time: datetime
    end_time: datetime
    description: str
    suggested_action: str
    affected_channels: List[str]


@dataclass
class QualityAssessment:
    """质量评估结果"""
    overall_score: float  # 总体质量分数 0-100
    quality_level: QualityLevel
    issues: List[QualityIssue]
    channel_scores: Dict[str, float]
    assessment_time: datetime
    metadata: Dict[str, Any]


class QualityAssessmentService:
    """质量评估服务"""
    
    def __init__(self):
        self.assessment_history = []
    
    async def assess_vibration_quality(self, 
                                     vibration_data: pd.DataFrame,
                                     config: Dict[str, Any]) -> QualityAssessment:
        """评估振动数据质量"""
        try:
            logger.info("开始振动数据质量评估...")
            
            # 1. 基础质量检查
            basic_issues = self._check_basic_quality(vibration_data, config)
            
            # 2. 波形连续性检查
            continuity_issues = self._check_waveform_continuity(vibration_data, config)
            
            # 3. 噪声水平检查
            noise_issues = self._check_noise_level(vibration_data, config)
            
            # 4. 数据完整性检查
            completeness_issues = self._check_data_completeness(vibration_data, config)
            
            # 5. 振幅异常检查
            amplitude_issues = self._check_amplitude_anomalies(vibration_data, config)
            
            # 6. 频率稳定性检查
            frequency_issues = self._check_frequency_stability(vibration_data, config)
            
            # 合并所有问题
            all_issues = (basic_issues + continuity_issues + noise_issues + 
                         completeness_issues + amplitude_issues + frequency_issues)
            
            # 计算质量分数
            overall_score = self._calculate_quality_score(all_issues, vibration_data)
            quality_level = self._determine_quality_level(overall_score)
            
            # 计算各通道质量分数
            channel_scores = self._calculate_channel_scores(vibration_data, all_issues)
            
            assessment = QualityAssessment(
                overall_score=overall_score,
                quality_level=quality_level,
                issues=all_issues,
                channel_scores=channel_scores,
                assessment_time=datetime.now(),
                metadata={
                    'data_shape': vibration_data.shape,
                    'sampling_rate': config.get('sampling_rate', 1000),
                    'assessment_config': config
                }
            )
            
            self.assessment_history.append(assessment)
            logger.info(f"质量评估完成，总体分数: {overall_score:.2f}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            raise
    
    def _check_basic_quality(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """基础质量检查"""
        issues = []
        
        # 检查数据类型
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            if not np.issubdtype(data[col].dtype, np.number):
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.ABNORMAL_AMPLITUDE,
                    severity=0.8,
                    start_time=data.index[0] if hasattr(data.index[0], 'to_pydatetime') else datetime.now(),
                    end_time=data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now(),
                    description=f"列 {col} 数据类型异常",
                    suggested_action="检查数据源，确保数据类型正确",
                    affected_channels=[col]
                ))
        
        return issues
    
    def _check_waveform_continuity(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """检查波形连续性"""
        issues = []
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal = data[col].values
            
            # 检查数据跳跃
            diff = np.diff(signal)
            jump_threshold = config.get('jump_threshold', 5.0)
            jump_indices = np.where(np.abs(diff) > jump_threshold)[0]
            
            if len(jump_indices) > 0:
                # 找到跳跃段
                jump_segments = self._find_continuous_segments(jump_indices)
                
                for start_idx, end_idx in jump_segments:
                    start_time = data.index[start_idx] if hasattr(data.index[start_idx], 'to_pydatetime') else datetime.now()
                    end_time = data.index[end_idx] if hasattr(data.index[end_idx], 'to_pydatetime') else datetime.now()
                    
                    severity = min(1.0, len(jump_segments) / 10.0)  # 跳跃段越多，严重程度越高
                    
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.DISCONTINUOUS_WAVEFORM,
                        severity=severity,
                        start_time=start_time,
                        end_time=end_time,
                        description=f"通道 {col} 在时间段 {start_time} - {end_time} 存在波形跳跃",
                        suggested_action="应用低通滤波器或重新采集数据",
                        affected_channels=[col]
                    ))
        
        return issues
    
    def _check_noise_level(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """检查噪声水平"""
        issues = []
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal = data[col].values
            
            # 计算信噪比
            signal_power = np.mean(signal**2)
            noise_power = np.var(signal - np.mean(signal))
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            # 检查信噪比
            min_snr = config.get('min_snr', 20.0)  # 最小信噪比阈值
            
            if snr < min_snr:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.NOISE_POLLUTION,
                    severity=min(1.0, (min_snr - snr) / min_snr),
                    start_time=data.index[0] if hasattr(data.index[0], 'to_pydatetime') else datetime.now(),
                    end_time=data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now(),
                    description=f"通道 {col} 信噪比过低 ({snr:.2f} dB)",
                    suggested_action="应用带通滤波器或改善传感器安装",
                    affected_channels=[col]
                ))
        
        return issues
    
    def _check_data_completeness(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """检查数据完整性"""
        issues = []
        
        # 检查缺失值
        missing_data = data.isnull().sum()
        total_rows = len(data)
        
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                missing_ratio = missing_count / total_rows
                
                if missing_ratio > config.get('max_missing_ratio', 0.05):  # 5%缺失率阈值
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.MISSING_DATA,
                        severity=missing_ratio,
                        start_time=data.index[0] if hasattr(data.index[0], 'to_pydatetime') else datetime.now(),
                        end_time=data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now(),
                        description=f"通道 {col} 缺失数据比例 {missing_ratio:.2%}",
                        suggested_action="插值处理或重新采集数据",
                        affected_channels=[col]
                    ))
        
        return issues
    
    def _check_amplitude_anomalies(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """检查振幅异常"""
        issues = []
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal = data[col].values
            
            # 检查异常振幅
            mean_amp = np.mean(np.abs(signal))
            std_amp = np.std(np.abs(signal))
            
            # 使用3-sigma规则检测异常
            threshold = mean_amp + 3 * std_amp
            anomaly_indices = np.where(np.abs(signal) > threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_segments = self._find_continuous_segments(anomaly_indices)
                
                for start_idx, end_idx in anomaly_segments:
                    start_time = data.index[start_idx] if hasattr(data.index[start_idx], 'to_pydatetime') else datetime.now()
                    end_time = data.index[end_idx] if hasattr(data.index[end_idx], 'to_pydatetime') else datetime.now()
                    
                    severity = min(1.0, len(anomaly_segments) / 5.0)
                    
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.ABNORMAL_AMPLITUDE,
                        severity=severity,
                        start_time=start_time,
                        end_time=end_time,
                        description=f"通道 {col} 在时间段 {start_time} - {end_time} 存在异常振幅",
                        suggested_action="应用振幅限制或检查传感器状态",
                        affected_channels=[col]
                    ))
        
        return issues
    
    def _check_frequency_stability(self, data: pd.DataFrame, config: Dict[str, Any]) -> List[QualityIssue]:
        """检查频率稳定性"""
        issues = []
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        sampling_rate = config.get('sampling_rate', 1000)
        
        for col in vibration_cols:
            signal = data[col].values
            
            # 使用滑动窗口计算频率稳定性
            window_size = config.get('frequency_window_size', 1000)
            if len(signal) < window_size * 2:
                continue
            
            frequencies = []
            for i in range(0, len(signal) - window_size, window_size // 2):
                window = signal[i:i + window_size]
                # 简单的频率估计
                fft_result = np.fft.fft(window)
                freqs = np.fft.fftfreq(len(window), 1/sampling_rate)
                dominant_freq = freqs[np.argmax(np.abs(fft_result[1:len(fft_result)//2])) + 1]
                frequencies.append(dominant_freq)
            
            if len(frequencies) > 1:
                freq_std = np.std(frequencies)
                freq_mean = np.mean(frequencies)
                
                # 检查频率变化
                max_freq_change = config.get('max_freq_change', 0.1)  # 10%频率变化阈值
                if freq_std / freq_mean > max_freq_change:
                    issues.append(QualityIssue(
                        issue_type=QualityIssueType.FREQUENCY_DRIFT,
                        severity=min(1.0, (freq_std / freq_mean) / max_freq_change),
                        start_time=data.index[0] if hasattr(data.index[0], 'to_pydatetime') else datetime.now(),
                        end_time=data.index[-1] if hasattr(data.index[-1], 'to_pydatetime') else datetime.now(),
                        description=f"通道 {col} 频率不稳定，标准差 {freq_std:.2f} Hz",
                        suggested_action="检查设备转速稳定性或应用频率校正",
                        affected_channels=[col]
                    ))
        
        return issues
    
    def _find_continuous_segments(self, indices: np.ndarray) -> List[Tuple[int, int]]:
        """找到连续的段"""
        if len(indices) == 0:
            return []
        
        segments = []
        start_idx = indices[0]
        prev_idx = indices[0]
        
        for idx in indices[1:]:
            if idx - prev_idx > 1:  # 不连续
                segments.append((start_idx, prev_idx))
                start_idx = idx
            prev_idx = idx
        
        segments.append((start_idx, prev_idx))
        return segments
    
    def _calculate_quality_score(self, issues: List[QualityIssue], data: pd.DataFrame) -> float:
        """计算质量分数"""
        if not issues:
            return 100.0
        
        # 基础分数
        base_score = 100.0
        
        # 根据问题严重程度扣分
        total_penalty = 0.0
        for issue in issues:
            # 严重程度越高，扣分越多
            penalty = issue.severity * 20.0  # 每个问题最多扣20分
            total_penalty += penalty
        
        # 考虑数据量，数据量越大，容错性越好
        data_factor = min(1.0, len(data) / 10000.0)  # 数据量因子
        adjusted_penalty = total_penalty * (1.0 - data_factor * 0.3)
        
        final_score = max(0.0, base_score - adjusted_penalty)
        return final_score
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.FAIR
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _calculate_channel_scores(self, data: pd.DataFrame, issues: List[QualityIssue]) -> Dict[str, float]:
        """计算各通道质量分数"""
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        channel_scores = {}
        
        for col in vibration_cols:
            # 基础分数
            base_score = 100.0
            
            # 计算该通道的问题
            channel_issues = [issue for issue in issues if col in issue.affected_channels]
            
            # 扣分
            total_penalty = sum(issue.severity * 15.0 for issue in channel_issues)
            channel_scores[col] = max(0.0, base_score - total_penalty)
        
        return channel_scores
    
    async def get_assessment_history(self) -> List[QualityAssessment]:
        """获取评估历史"""
        return self.assessment_history
    
    async def get_assessment_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.assessment_history:
            return {}
        
        recent_assessments = self.assessment_history[-10:]  # 最近10次评估
        
        return {
            'total_assessments': len(self.assessment_history),
            'recent_average_score': np.mean([a.overall_score for a in recent_assessments]),
            'quality_level_distribution': {
                level.value: len([a for a in self.assessment_history if a.quality_level == level])
                for level in QualityLevel
            },
            'common_issues': self._get_common_issues(),
            'last_assessment_time': self.assessment_history[-1].assessment_time
        }
    
    def _get_common_issues(self) -> Dict[str, int]:
        """获取常见问题统计"""
        issue_counts = {}
        for assessment in self.assessment_history:
            for issue in assessment.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return issue_counts 