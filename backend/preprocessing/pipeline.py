 """
预处理管道服务
根据质量评估结果进行数据重新预处理
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class PreprocessingType(Enum):
    """预处理类型"""
    LOWPASS_FILTER = "lowpass_filter"  # 低通滤波
    HIGHPASS_FILTER = "highpass_filter"  # 高通滤波
    BANDPASS_FILTER = "bandpass_filter"  # 带通滤波
    NOTCH_FILTER = "notch_filter"  # 陷波滤波
    AMPLITUDE_LIMITING = "amplitude_limiting"  # 振幅限制
    NOISE_REDUCTION = "noise_reduction"  # 噪声抑制
    DATA_INTERPOLATION = "data_interpolation"  # 数据插值
    PHASE_CORRECTION = "phase_correction"  # 相位校正
    FREQUENCY_CORRECTION = "frequency_correction"  # 频率校正


@dataclass
class PreprocessingStep:
    """预处理步骤"""
    step_type: PreprocessingType
    parameters: Dict[str, Any]
    description: str
    applied_channels: List[str]
    timestamp: datetime


@dataclass
class PreprocessingResult:
    """预处理结果"""
    processed_data: pd.DataFrame
    original_data: pd.DataFrame
    steps: List[PreprocessingStep]
    quality_improvement: Dict[str, float]  # 各通道质量改善程度
    processing_time: float
    metadata: Dict[str, Any]


class PreprocessingPipeline:
    """预处理管道"""
    
    def __init__(self):
        self.processing_history = []
    
    async def process_data(self, 
                          data: pd.DataFrame,
                          quality_issues: List[Any],  # QualityIssue类型
                          config: Dict[str, Any]) -> PreprocessingResult:
        """根据质量问题进行数据预处理"""
        try:
            logger.info("开始数据预处理...")
            start_time = datetime.now()
            
            processed_data = data.copy()
            steps = []
            
            # 按问题类型分组处理
            issue_groups = self._group_issues_by_type(quality_issues)
            
            # 1. 处理波形不连续问题
            if 'discontinuous_waveform' in issue_groups:
                processed_data, step = await self._handle_discontinuous_waveform(
                    processed_data, issue_groups['discontinuous_waveform'], config
                )
                steps.append(step)
            
            # 2. 处理噪声污染问题
            if 'noise_pollution' in issue_groups:
                processed_data, step = await self._handle_noise_pollution(
                    processed_data, issue_groups['noise_pollution'], config
                )
                steps.append(step)
            
            # 3. 处理数据缺失问题
            if 'missing_data' in issue_groups:
                processed_data, step = await self._handle_missing_data(
                    processed_data, issue_groups['missing_data'], config
                )
                steps.append(step)
            
            # 4. 处理异常振幅问题
            if 'abnormal_amplitude' in issue_groups:
                processed_data, step = await self._handle_abnormal_amplitude(
                    processed_data, issue_groups['abnormal_amplitude'], config
                )
                steps.append(step)
            
            # 5. 处理频率漂移问题
            if 'frequency_drift' in issue_groups:
                processed_data, step = await self._handle_frequency_drift(
                    processed_data, issue_groups['frequency_drift'], config
                )
                steps.append(step)
            
            # 6. 处理相位偏移问题
            if 'phase_shift' in issue_groups:
                processed_data, step = await self._handle_phase_shift(
                    processed_data, issue_groups['phase_shift'], config
                )
                steps.append(step)
            
            # 计算质量改善程度
            quality_improvement = await self._calculate_quality_improvement(
                data, processed_data, quality_issues
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = PreprocessingResult(
                processed_data=processed_data,
                original_data=data,
                steps=steps,
                quality_improvement=quality_improvement,
                processing_time=processing_time,
                metadata={
                    'original_shape': data.shape,
                    'processed_shape': processed_data.shape,
                    'config': config
                }
            )
            
            self.processing_history.append(result)
            logger.info(f"数据预处理完成，耗时: {processing_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def _group_issues_by_type(self, issues: List[Any]) -> Dict[str, List[Any]]:
        """按问题类型分组"""
        groups = {}
        for issue in issues:
            issue_type = issue.issue_type.value
            if issue_type not in groups:
                groups[issue_type] = []
            groups[issue_type].append(issue)
        return groups
    
    async def _handle_discontinuous_waveform(self, 
                                           data: pd.DataFrame,
                                           issues: List[Any],
                                           config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理波形不连续问题"""
        logger.info("处理波形不连续问题...")
        
        # 获取受影响的通道
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        
        for channel in affected_channels:
            if channel in data.columns:
                signal_data = data[channel].values
                
                # 应用低通滤波器平滑波形
                cutoff_freq = config.get('discontinuity_cutoff_freq', 100)  # Hz
                sampling_rate = config.get('sampling_rate', 1000)
                
                # 设计低通滤波器
                nyquist = sampling_rate / 2
                cutoff_norm = cutoff_freq / nyquist
                b, a = signal.butter(4, cutoff_norm, btype='low')
                
                # 应用滤波器
                filtered_signal = signal.filtfilt(b, a, signal_data)
                processed_data[channel] = filtered_signal
        
        step = PreprocessingStep(
            step_type=PreprocessingType.LOWPASS_FILTER,
            parameters={
                'cutoff_freq': cutoff_freq,
                'filter_order': 4,
                'filter_type': 'butterworth'
            },
            description="应用低通滤波器处理波形不连续",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _handle_noise_pollution(self, 
                                    data: pd.DataFrame,
                                    issues: List[Any],
                                    config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理噪声污染问题"""
        logger.info("处理噪声污染问题...")
        
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        
        for channel in affected_channels:
            if channel in data.columns:
                signal_data = data[channel].values
                
                # 应用带通滤波器去除噪声
                low_cutoff = config.get('noise_low_cutoff', 10)  # Hz
                high_cutoff = config.get('noise_high_cutoff', 500)  # Hz
                sampling_rate = config.get('sampling_rate', 1000)
                
                # 设计带通滤波器
                nyquist = sampling_rate / 2
                low_norm = low_cutoff / nyquist
                high_norm = high_cutoff / nyquist
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                
                # 应用滤波器
                filtered_signal = signal.filtfilt(b, a, signal_data)
                processed_data[channel] = filtered_signal
        
        step = PreprocessingStep(
            step_type=PreprocessingType.BANDPASS_FILTER,
            parameters={
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff,
                'filter_order': 4,
                'filter_type': 'butterworth'
            },
            description="应用带通滤波器去除噪声",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _handle_missing_data(self, 
                                 data: pd.DataFrame,
                                 issues: List[Any],
                                 config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理数据缺失问题"""
        logger.info("处理数据缺失问题...")
        
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        
        for channel in affected_channels:
            if channel in data.columns:
                # 检查缺失值
                missing_mask = data[channel].isnull()
                
                if missing_mask.any():
                    # 使用线性插值填充缺失值
                    valid_indices = ~missing_mask
                    valid_data = data[channel][valid_indices]
                    
                    if len(valid_data) > 1:
                        # 创建插值函数
                        interp_func = interp1d(
                            np.arange(len(data))[valid_indices],
                            valid_data,
                            kind='linear',
                            bounds_error=False,
                            fill_value='extrapolate'
                        )
                        
                        # 插值填充
                        interpolated_values = interp_func(np.arange(len(data)))
                        processed_data[channel] = interpolated_values
        
        step = PreprocessingStep(
            step_type=PreprocessingType.DATA_INTERPOLATION,
            parameters={
                'interpolation_method': 'linear',
                'fill_method': 'extrapolate'
            },
            description="使用线性插值填充缺失数据",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _handle_abnormal_amplitude(self, 
                                       data: pd.DataFrame,
                                       issues: List[Any],
                                       config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理异常振幅问题"""
        logger.info("处理异常振幅问题...")
        
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        
        for channel in affected_channels:
            if channel in data.columns:
                signal_data = data[channel].values
                
                # 计算振幅限制阈值
                mean_amp = np.mean(np.abs(signal_data))
                std_amp = np.std(np.abs(signal_data))
                limit_threshold = mean_amp + 2 * std_amp  # 2-sigma限制
                
                # 应用振幅限制
                limited_signal = np.clip(signal_data, -limit_threshold, limit_threshold)
                processed_data[channel] = limited_signal
        
        step = PreprocessingStep(
            step_type=PreprocessingType.AMPLITUDE_LIMITING,
            parameters={
                'limit_method': '2_sigma',
                'threshold_factor': 2.0
            },
            description="应用振幅限制处理异常值",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _handle_frequency_drift(self, 
                                    data: pd.DataFrame,
                                    issues: List[Any],
                                    config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理频率漂移问题"""
        logger.info("处理频率漂移问题...")
        
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        sampling_rate = config.get('sampling_rate', 1000)
        
        for channel in affected_channels:
            if channel in data.columns:
                signal_data = data[channel].values
                
                # 使用自适应滤波器校正频率漂移
                # 这里使用简单的滑动平均作为示例
                window_size = config.get('frequency_correction_window', 100)
                
                if len(signal_data) > window_size:
                    # 应用滑动平均平滑频率变化
                    corrected_signal = np.convolve(
                        signal_data, 
                        np.ones(window_size) / window_size, 
                        mode='same'
                    )
                    processed_data[channel] = corrected_signal
        
        step = PreprocessingStep(
            step_type=PreprocessingType.FREQUENCY_CORRECTION,
            parameters={
                'correction_method': 'sliding_average',
                'window_size': window_size
            },
            description="使用滑动平均校正频率漂移",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _handle_phase_shift(self, 
                                data: pd.DataFrame,
                                issues: List[Any],
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, PreprocessingStep]:
        """处理相位偏移问题"""
        logger.info("处理相位偏移问题...")
        
        affected_channels = set()
        for issue in issues:
            affected_channels.update(issue.affected_channels)
        
        processed_data = data.copy()
        
        for channel in affected_channels:
            if channel in data.columns:
                signal_data = data[channel].values
                
                # 使用希尔伯特变换进行相位校正
                # 这里使用简单的相位对齐作为示例
                # 实际应用中可能需要更复杂的相位校正算法
                
                # 计算信号的相位
                analytic_signal = signal.hilbert(signal_data)
                phase = np.angle(analytic_signal)
                
                # 简单的相位校正（去除直流分量）
                corrected_phase = phase - np.mean(phase)
                
                # 重建信号
                amplitude = np.abs(analytic_signal)
                corrected_signal = amplitude * np.cos(corrected_phase)
                processed_data[channel] = corrected_signal
        
        step = PreprocessingStep(
            step_type=PreprocessingType.PHASE_CORRECTION,
            parameters={
                'correction_method': 'hilbert_transform',
                'remove_dc': True
            },
            description="使用希尔伯特变换进行相位校正",
            applied_channels=list(affected_channels),
            timestamp=datetime.now()
        )
        
        return processed_data, step
    
    async def _calculate_quality_improvement(self, 
                                           original_data: pd.DataFrame,
                                           processed_data: pd.DataFrame,
                                           original_issues: List[Any]) -> Dict[str, float]:
        """计算质量改善程度"""
        improvement = {}
        
        # 获取所有振动通道
        vibration_cols = [col for col in original_data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for channel in vibration_cols:
            if channel in original_data.columns and channel in processed_data.columns:
                original_signal = original_data[channel].values
                processed_signal = processed_data[channel].values
                
                # 计算信噪比改善
                original_snr = self._calculate_snr(original_signal)
                processed_snr = self._calculate_snr(processed_signal)
                
                if original_snr > 0:
                    snr_improvement = (processed_snr - original_snr) / original_snr * 100
                else:
                    snr_improvement = 0
                
                # 计算平滑度改善
                original_smoothness = self._calculate_smoothness(original_signal)
                processed_smoothness = self._calculate_smoothness(processed_signal)
                
                if original_smoothness > 0:
                    smoothness_improvement = (processed_smoothness - original_smoothness) / original_smoothness * 100
                else:
                    smoothness_improvement = 0
                
                # 综合改善分数
                improvement[channel] = (snr_improvement + smoothness_improvement) / 2
        
        return improvement
    
    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        """计算信噪比"""
        signal_power = np.mean(signal_data**2)
        noise_power = np.var(signal_data - np.mean(signal_data))
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        else:
            return float('inf')
    
    def _calculate_smoothness(self, signal_data: np.ndarray) -> float:
        """计算信号平滑度（基于二阶差分）"""
        if len(signal_data) < 3:
            return 0.0
        
        # 计算二阶差分
        second_diff = np.diff(signal_data, n=2)
        
        # 平滑度 = 1 / (1 + 二阶差分的方差)
        smoothness = 1.0 / (1.0 + np.var(second_diff))
        return smoothness
    
    async def get_processing_history(self) -> List[PreprocessingResult]:
        """获取处理历史"""
        return self.processing_history
    
    async def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        if not self.processing_history:
            return {}
        
        recent_results = self.processing_history[-10:]  # 最近10次处理
        
        return {
            'total_processings': len(self.processing_history),
            'average_processing_time': np.mean([r.processing_time for r in recent_results]),
            'average_quality_improvement': np.mean([
                np.mean(list(r.quality_improvement.values())) for r in recent_results
            ]),
            'most_common_steps': self._get_most_common_steps(),
            'last_processing_time': self.processing_history[-1].metadata.get('timestamp', datetime.now())
        }
    
    def _get_most_common_steps(self) -> Dict[str, int]:
        """获取最常见的处理步骤"""
        step_counts = {}
        for result in self.processing_history:
            for step in result.steps:
                step_type = step.step_type.value
                step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        return step_counts