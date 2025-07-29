## 振动算法交互式训练参数设计

### 1. **基础算法参数**
```json
{
  "algorithm_config": {
    "function": "vibrate31",
    "data_type": "float32",
    "sampling_rate": 1000,
    "speed_unit": "rpm"
  }
}
```

### 2. **传感器配置参数**
```json
{
  "sensor_config": {
    "sensor_type": "accelerometer",
    "sensor_range": "±10g",
    "sensitivity": 100,
    "mounting_position": "horizontal"
  }
}
```

### 3. **振动算法特定参数**
```json
{
  "vibrate_specific": {
    "duration_limit": 5,
    "dc_threshold": null,
    "custom_param": {
      "vibration_analysis_type": "fft",
      "window_function": "hanning",
      "overlap_ratio": 0.5,
      "frequency_resolution": 0.1
    }
  }
}
```

### 4. **交互式调试参数**
```json
{
  "interactive_debug": {
    "frequency_filtering": {
      "enabled": true,
      "low_freq_cutoff": 10,
      "high_freq_cutoff": 1000,
      "bandpass_filters": [
        {"name": "bearing_freq", "center": 50, "bandwidth": 10},
        {"name": "gear_freq", "center": 200, "bandwidth": 20}
      ]
    },
    "amplitude_thresholds": {
      "warning_level": 0.5,
      "alarm_level": 1.0,
      "critical_level": 2.0
    },
    "time_domain_features": {
      "rms_enabled": true,
      "peak_enabled": true,
      "crest_factor_enabled": true,
      "kurtosis_enabled": true
    },
    "frequency_domain_features": {
      "spectrum_enabled": true,
      "harmonic_analysis": true,
      "sideband_analysis": true,
      "envelope_analysis": true
    },
    "data_selection": {
      "time_range": {
        "start_time": "2024-01-01 00:00:00",
        "end_time": "2024-01-01 23:59:59"
      },
      "speed_range": {
        "min_speed": 800,
        "max_speed": 1200
      },
      "quality_filters": {
        "min_signal_quality": 0.8,
        "max_noise_level": 0.1
      }
    }
  }
}
```

### 5. **实时调整参数**
```json
{
  "real_time_adjustment": {
    "adaptive_thresholds": {
      "enabled": true,
      "learning_rate": 0.1,
      "update_frequency": "hourly"
    },
    "dynamic_filtering": {
      "enabled": true,
      "auto_adjust_bandwidth": true,
      "noise_adaptation": true
    },
    "feature_weights": {
      "rms_weight": 0.3,
      "peak_weight": 0.2,
      "crest_factor_weight": 0.2,
      "kurtosis_weight": 0.3
    }
  }
}
```

## 振动算法交互式训练流程

### 1. **数据预处理阶段**
- **跳过传统预处理**: 振动算法直接使用原始波形数据
- **实时数据流**: 支持实时数据输入和在线分析
- **质量评估**: 自动评估信号质量，过滤低质量数据

### 2. **特征提取阶段**
- **时域特征**: RMS、峰值、峰值因子、峭度等
- **频域特征**: 频谱分析、谐波分析、边带分析
- **包络分析**: 用于轴承故障检测
- **自适应特征**: 根据设备类型自动调整特征集

### 3. **交互式调试阶段**
- **频率范围调整**: 用户可以在频谱图上选择关注的频率范围
- **阈值设置**: 通过可视化界面调整警告、报警、危险阈值
- **特征权重调整**: 调整不同特征的重要性权重
- **实时预览**: 立即看到参数调整对分析结果的影响

### 4. **模型训练阶段**
- **增量学习**: 支持在线学习和模型更新
- **多模型融合**: 结合多个振动分析模型的结果
- **自适应优化**: 根据设备运行状态自动优化参数

## 与状态识别算法的区别

### 振动算法特点：
1. **无数据预处理**: 直接使用原始振动信号
2. **频域分析**: 重点关注频谱特征
3. **实时性**: 支持实时监控和在线分析
4. **设备特定**: 参数配置与具体设备类型相关

### 状态识别算法特点：
1. **需要预处理**: 异常值处理、特征选择等
2. **特征工程**: 复杂的特征变换和选择
3. **批量处理**: 主要用于离线分析
4. **通用性**: 适用于多种设备类型