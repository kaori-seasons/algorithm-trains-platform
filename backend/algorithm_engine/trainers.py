"""
算法训练器模块
实现各种算法类型的训练逻辑
"""
import asyncio
import logging
import json
import pickle
import os
import zipfile
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .models import TrainingConfig, TrainingResult, AlgorithmType, TrainingStatus

logger = logging.getLogger(__name__)


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.training_history = []
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练模型"""
        raise NotImplementedError("子类必须实现train方法")
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成参数"""
        raise NotImplementedError("子类必须实现generate_parameters方法")
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据"""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            return pd.read_json(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
    
    def _save_model(self, model, output_path: str, model_format: str):
        """保存模型"""
        if model_format == 'json':
            # 对于sklearn模型，保存为joblib格式
            joblib.dump(model, f"{output_path}/model.joblib")
            # 同时保存JSON格式的参数
            model_params = self._extract_model_params(model)
            with open(f"{output_path}/parameters.json", 'w') as f:
                json.dump(model_params, f, indent=2)
        elif model_format == 'pickle':
            with open(f"{output_path}/model.pkl", 'wb') as f:
                pickle.dump(model, f)
        elif model_format == 'm':
            # .m文件实际上是pickle格式，兼容MATLAB命名习惯
            with open(f"{output_path}/model.m", 'wb') as f:
                pickle.dump(model, f)
            
            # 同时保存参数文件
            model_params = self._extract_model_params(model)
            with open(f"{output_path}/parameters.json", 'w', encoding='utf-8') as f:
                json.dump(model_params, f, indent=2, ensure_ascii=False)
        elif model_format == 'zip':
            # 创建临时目录
            temp_dir = f"{output_path}/temp_model"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存模型文件
            joblib.dump(model, f"{temp_dir}/model.joblib")
            
            # 保存参数文件
            model_params = self._extract_model_params(model)
            with open(f"{temp_dir}/parameters.json", 'w', encoding='utf-8') as f:
                json.dump(model_params, f, indent=2, ensure_ascii=False)
            
            # 创建压缩包
            zip_path = f"{output_path}/model.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_dir)
        else:
            raise ValueError(f"不支持的模型格式: {model_format}")
    
    def _extract_model_params(self, model) -> Dict[str, Any]:
        """提取模型参数"""
        if hasattr(model, 'get_params'):
            return model.get_params()
        return {}
    
    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }


class StatusRecognitionTrainer(BaseTrainer):
    """状态识别算法训练器 - 支持不依赖于模型训练的交互式调试"""
    
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.training_history = []
        self.visualization_data = {}
        self.interactive_params = {}
        self.raw_data = None
        self.processed_data = None
        self.data_analysis = {}
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练状态识别模型 - 支持交互式调试"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            train_data = self._load_data(config.train_data_path)
            self.raw_data = train_data.copy()
            
            # 准备特征和目标
            X = train_data[config.feature_columns]
            y = train_data[config.target_column]
            
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和验证数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 获取模型类型和参数
            model_type = config.algorithm_params.get('model_type', 'random_forest')
            model_params = config.algorithm_params.get('model_parameters', {})
            
            # 创建模型
            self.current_model = self._create_model(model_type, model_params)
            
            # 训练模型
            self.current_model.fit(X_train, y_train)
            
            # 生成数据分析和可视化数据
            self.data_analysis = self._analyze_data(X, y, config.feature_columns)
            self.visualization_data = self._generate_visualization_data(
                self.current_model, X_val, y_val, X_train, y_train
            )
            
            # 预测和评估
            y_pred = self.current_model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # 保存模型
            if config.save_model:
                self._save_model(self.current_model, config.output_path, config.model_format.value)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TrainingResult(
                task_id=f"status_recognition_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.STATUS_RECOGNITION,
                status=TrainingStatus.COMPLETED,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/model.joblib",
                parameters_path=f"{config.output_path}/parameters.json",
                metadata={
                    'feature_columns': config.feature_columns,
                    'target_column': config.target_column,
                    'n_samples': len(train_data),
                    'model_type': model_type,
                    'visualization_data': self.visualization_data,
                    'data_analysis': self.data_analysis,
                    'interactive_params': self.interactive_params
                }
            )
            
        except Exception as e:
            logger.error(f"状态识别模型训练失败: {str(e)}")
            raise
    
    def _analyze_data(self, X: pd.DataFrame, y: pd.Series, feature_columns: List[str]) -> Dict[str, Any]:
        """分析数据特征，为交互式调试提供基础"""
        analysis = {}
        
        # 1. 数据分布分析
        analysis['data_distribution'] = {}
        for col in feature_columns:
            analysis['data_distribution'][col] = {
                'mean': float(X[col].mean()),
                'std': float(X[col].std()),
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'q25': float(X[col].quantile(0.25)),
                'q50': float(X[col].quantile(0.50)),
                'q75': float(X[col].quantile(0.75))
            }
        
        # 2. 目标变量分布
        analysis['target_distribution'] = {
            'class_counts': y.value_counts().to_dict(),
            'class_balance': len(y[y == 1]) / len(y)
        }
        
        # 3. 特征与目标的关系分析
        analysis['feature_target_correlation'] = {}
        for col in feature_columns:
            correlation = X[col].corr(y)
            analysis['feature_target_correlation'][col] = float(correlation)
        
        # 4. 异常值检测
        analysis['outliers'] = {}
        for col in feature_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
            analysis['outliers'][col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(X) * 100,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        # 5. 数据质量分析
        analysis['data_quality'] = {
            'missing_values': X.isnull().sum().to_dict(),
            'duplicate_rows': len(X[X.duplicated()]),
            'total_rows': len(X)
        }
        
        return analysis
    
    async def interactive_debug(self, debug_params: Dict[str, Any]) -> Dict[str, Any]:
        """交互式调试 - 不依赖模型训练，直接基于数据分析"""
        try:
            if self.raw_data is None:
                raise ValueError("请先运行训练以加载数据")
            
            # 应用调试参数
            processed_data = self._apply_debug_parameters(self.raw_data, debug_params)
            
            # 生成调试结果
            debug_results = self._generate_debug_results(processed_data, debug_params)
            
            # 更新交互式参数
            self.interactive_params.update(debug_params)
            
            return {
                "success": True,
                "debug_params": debug_params,
                "results": debug_results,
                "data_summary": {
                    "original_rows": len(self.raw_data),
                    "processed_rows": len(processed_data),
                    "removed_rows": len(self.raw_data) - len(processed_data)
                }
            }
            
        except Exception as e:
            logger.error(f"交互式调试失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _apply_debug_parameters(self, data: pd.DataFrame, debug_params: Dict[str, Any]) -> pd.DataFrame:
        """应用调试参数到数据"""
        processed_data = data.copy()
        
        # 1. 数据过滤
        if 'data_filters' in debug_params:
            for filter_condition in debug_params['data_filters']:
                mask = self._apply_filter_condition(processed_data, filter_condition)
                processed_data = processed_data[mask]
        
        # 2. 异常值处理
        if 'outlier_handling' in debug_params:
            outlier_config = debug_params['outlier_handling']
            for col in outlier_config.get('columns', []):
                if col in processed_data.columns:
                    method = outlier_config.get('method', 'iqr')
                    if method == 'iqr':
                        processed_data = self._handle_outliers_iqr(processed_data, col, outlier_config)
                    elif method == 'zscore':
                        processed_data = self._handle_outliers_zscore(processed_data, col, outlier_config)
        
        # 3. 特征变换
        if 'feature_transformations' in debug_params:
            for transform in debug_params['feature_transformations']:
                col = transform.get('column')
                method = transform.get('method')
                if col in processed_data.columns:
                    if method == 'log':
                        processed_data[col] = np.log1p(processed_data[col])
                    elif method == 'sqrt':
                        processed_data[col] = np.sqrt(processed_data[col])
                    elif method == 'standardize':
                        processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
        
        # 4. 特征选择
        if 'feature_selection' in debug_params:
            selected_features = debug_params['feature_selection']
            if selected_features:
                # 确保目标列始终包含
                target_col = self.interactive_params.get('target_column', 'status')
                if target_col not in selected_features:
                    selected_features.append(target_col)
                processed_data = processed_data[selected_features]
        
        # 5. 数据采样
        if 'sampling' in debug_params:
            sampling_config = debug_params['sampling']
            method = sampling_config.get('method', 'random')
            size = sampling_config.get('size', 1000)
            
            if method == 'random':
                processed_data = processed_data.sample(n=min(size, len(processed_data)), random_state=42)
            elif method == 'stratified':
                target_col = self.interactive_params.get('target_column', 'status')
                if target_col in processed_data.columns:
                    processed_data = self._stratified_sample(processed_data, target_col, size)
        
        return processed_data
    
    def _handle_outliers_iqr(self, data: pd.DataFrame, column: str, config: Dict[str, Any]) -> pd.DataFrame:
        """使用IQR方法处理异常值"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        multiplier = config.get('multiplier', 1.5)
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        action = config.get('action', 'remove')
        if action == 'remove':
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        elif action == 'cap':
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
            return data
        
        return data
    
    def _handle_outliers_zscore(self, data: pd.DataFrame, column: str, config: Dict[str, Any]) -> pd.DataFrame:
        """使用Z-score方法处理异常值"""
        z_threshold = config.get('z_threshold', 3)
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        
        action = config.get('action', 'remove')
        if action == 'remove':
            return data[z_scores < z_threshold]
        elif action == 'cap':
            mean_val = data[column].mean()
            std_val = data[column].std()
            data[column] = data[column].clip(
                lower=mean_val - z_threshold * std_val,
                upper=mean_val + z_threshold * std_val
            )
            return data
        
        return data
    
    def _stratified_sample(self, data: pd.DataFrame, target_column: str, size: int) -> pd.DataFrame:
        """分层采样"""
        try:
            return data.groupby(target_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), size // len(data[target_column].unique()))
            )
        except:
            return data.sample(n=min(size, len(data)), random_state=42)
    
    def _generate_debug_results(self, processed_data: pd.DataFrame, debug_params: Dict[str, Any]) -> Dict[str, Any]:
        """生成调试结果"""
        results = {}
        
        # 1. 数据统计
        results['data_statistics'] = {
            'total_rows': len(processed_data),
            'total_columns': len(processed_data.columns),
            'memory_usage': processed_data.memory_usage(deep=True).sum()
        }
        
        # 2. 特征统计
        results['feature_statistics'] = {}
        for col in processed_data.columns:
            if processed_data[col].dtype in ['int64', 'float64']:
                results['feature_statistics'][col] = {
                    'mean': float(processed_data[col].mean()),
                    'std': float(processed_data[col].std()),
                    'min': float(processed_data[col].min()),
                    'max': float(processed_data[col].max())
                }
        
        # 3. 目标变量分析
        target_col = self.interactive_params.get('target_column', 'status')
        if target_col in processed_data.columns:
            results['target_analysis'] = {
                'class_distribution': processed_data[target_col].value_counts().to_dict(),
                'class_balance': len(processed_data[processed_data[target_col] == 1]) / len(processed_data)
            }
        
        # 4. 数据质量报告
        results['data_quality'] = {
            'missing_values': processed_data.isnull().sum().to_dict(),
            'duplicate_rows': len(processed_data[processed_data.duplicated()]),
            'unique_values': {col: processed_data[col].nunique() for col in processed_data.columns}
        }
        
        # 5. 可视化数据
        results['visualization_data'] = self._generate_debug_visualization(processed_data)
        
        return results
    
    def _generate_debug_visualization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成调试用的可视化数据"""
        viz_data = {}
        
        # 1. 特征分布直方图
        viz_data['feature_distributions'] = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                viz_data['feature_distributions'][col] = {
                    'values': data[col].tolist(),
                    'bins': np.histogram(data[col], bins=20)[0].tolist(),
                    'bin_edges': np.histogram(data[col], bins=20)[1].tolist()
                }
        
        # 2. 相关性热力图
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = data[numeric_cols].corr()
            viz_data['correlation_matrix'] = {
                'columns': numeric_cols.tolist(),
                'matrix': correlation_matrix.values.tolist()
            }
        
        # 3. 箱线图数据
        viz_data['boxplot_data'] = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                viz_data['boxplot_data'][col] = {
                    'q1': float(data[col].quantile(0.25)),
                    'q2': float(data[col].quantile(0.50)),
                    'q3': float(data[col].quantile(0.75)),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'outliers': data[col][(data[col] < data[col].quantile(0.25) - 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25))) | 
                                        (data[col] > data[col].quantile(0.75) + 1.5 * (data[col].quantile(0.75) - data[col].quantile(0.25)))].tolist()
                }
        
        return viz_data
    
    async def get_data_analysis(self) -> Dict[str, Any]:
        """获取数据分析结果"""
        return self.data_analysis
    
    async def get_debug_suggestions(self) -> Dict[str, Any]:
        """获取调试建议"""
        suggestions = {}
        
        if self.data_analysis:
            # 1. 异常值建议
            outlier_suggestions = []
            for col, outlier_info in self.data_analysis.get('outliers', {}).items():
                if outlier_info['percentage'] > 5:  # 异常值超过5%
                    outlier_suggestions.append({
                        'column': col,
                        'issue': f"异常值比例较高 ({outlier_info['percentage']:.1f}%)",
                        'suggestion': "考虑使用IQR或Z-score方法处理异常值",
                        'action': {
                            'type': 'outlier_handling',
                            'column': col,
                            'method': 'iqr',
                            'action': 'remove'
                        }
                    })
            suggestions['outlier_handling'] = outlier_suggestions
            
            # 2. 数据不平衡建议
            target_balance = self.data_analysis.get('target_distribution', {}).get('class_balance', 0.5)
            if target_balance < 0.3 or target_balance > 0.7:
                suggestions['class_imbalance'] = {
                    'issue': f"类别不平衡 (少数类比例: {target_balance:.1%})",
                    'suggestion': "考虑使用分层采样或调整类别权重",
                    'action': {
                        'type': 'sampling',
                        'method': 'stratified',
                        'size': 1000
                    }
                }
            
            # 3. 特征选择建议
            correlation_suggestions = []
            for col, corr in self.data_analysis.get('feature_target_correlation', {}).items():
                if abs(corr) < 0.1:  # 相关性很低
                    correlation_suggestions.append({
                        'column': col,
                        'correlation': corr,
                        'suggestion': "考虑移除该特征或进行特征变换"
                    })
            suggestions['feature_selection'] = correlation_suggestions
        
        return suggestions
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """创建指定类型的模型"""
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_type == 'svm':
            from sklearn.svm import SVC
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                gamma=params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=params.get('C', 1.0),
                penalty=params.get('penalty', 'l2'),
                solver=params.get('solver', 'lbfgs'),
                random_state=42
            )
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                random_state=42
            )
        elif model_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(
                n_neighbors=params.get('n_neighbors', 5),
                weights=params.get('weights', 'uniform'),
                algorithm=params.get('algorithm', 'auto')
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _generate_visualization_data(self, model, X_val, y_val, X_train, y_train) -> Dict[str, Any]:
        """生成用于交互式调参的可视化数据"""
        visualization_data = {}
        
        # 1. ROC曲线数据
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds = roc_curve(y_val, y_proba)
            roc_auc = auc(fpr, tpr)
            
            visualization_data['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc,
                'optimal_threshold': thresholds[np.argmax(tpr - fpr)]
            }
        
        # 2. 特征重要性数据
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            visualization_data['feature_importance'] = {
                'features': list(range(len(feature_importance))),
                'importance': feature_importance.tolist(),
                'sorted_indices': np.argsort(feature_importance)[::-1].tolist()
            }
        
        # 3. 学习曲线数据
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        visualization_data['learning_curve'] = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
            'val_scores_std': np.std(val_scores, axis=1).tolist()
        }
        
        # 4. 混淆矩阵数据
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        visualization_data['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': sorted(list(set(y_val)))
        }
        
        # 5. 参数敏感性分析数据
        if hasattr(model, 'get_params'):
            param_sensitivity = self._analyze_parameter_sensitivity(model, X_train, y_train, X_val, y_val)
            visualization_data['parameter_sensitivity'] = param_sensitivity
        
        return visualization_data
    
    def _analyze_parameter_sensitivity(self, model, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """分析参数敏感性"""
        sensitivity_data = {}
        base_params = model.get_params()
        
        # 分析关键参数
        key_params = ['n_estimators', 'max_depth', 'C', 'learning_rate', 'n_neighbors']
        
        for param in key_params:
            if param in base_params and base_params[param] is not None:
                param_values = self._generate_param_range(param, base_params[param])
                scores = []
                
                for value in param_values:
                    try:
                        # 创建新模型实例
                        model_copy = self._create_model_with_param(model, param, value)
                        model_copy.fit(X_train, y_train)
                        score = model_copy.score(X_val, y_val)
                        scores.append(score)
                    except:
                        scores.append(0.0)
                
                sensitivity_data[param] = {
                    'values': param_values,
                    'scores': scores,
                    'optimal_value': param_values[np.argmax(scores)]
                }
        
        return sensitivity_data
    
    def _generate_param_range(self, param: str, base_value) -> list:
        """生成参数范围"""
        if param == 'n_estimators':
            return [50, 100, 150, 200, 250, 300]
        elif param == 'max_depth':
            return [3, 5, 7, 10, 15, 20, None]
        elif param == 'C':
            return [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        elif param == 'learning_rate':
            return [0.01, 0.05, 0.1, 0.2, 0.3]
        elif param == 'n_neighbors':
            return [3, 5, 7, 9, 11, 15]
        else:
            return [base_value]
    
    def _create_model_with_param(self, model, param: str, value) -> Any:
        """创建具有特定参数值的模型副本"""
        model_type = type(model).__name__
        params = model.get_params()
        params[param] = value
        return self._create_model(model_type.lower().replace('classifier', ''), params)
    
    def _apply_filter_condition(self, X: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
        """应用过滤条件"""
        column = condition.get('column')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if column not in X.columns:
            return pd.Series([True] * len(X))
        
        if operator == '>':
            return X[column] > value
        elif operator == '<':
            return X[column] < value
        elif operator == '>=':
            return X[column] >= value
        elif operator == '<=':
            return X[column] <= value
        elif operator == '==':
            return X[column] == value
        elif operator == '!=':
            return X[column] != value
        elif operator == 'in':
            return X[column].isin(value)
        else:
            return pd.Series([True] * len(X))
    
    async def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据用于前端展示"""
        return self.visualization_data
    
    async def get_optimal_parameters(self) -> Dict[str, Any]:
        """获取基于可视化分析的最优参数建议"""
        optimal_params = {}
        
        if 'roc_curve' in self.visualization_data:
            optimal_params['classification_threshold'] = self.visualization_data['roc_curve']['optimal_threshold']
        
        if 'feature_importance' in self.visualization_data:
            # 选择重要性大于平均值的特征
            importance = self.visualization_data['feature_importance']['importance']
            mean_importance = np.mean(importance)
            selected_features = [i for i, imp in enumerate(importance) if imp > mean_importance]
            optimal_params['selected_features'] = selected_features
        
        if 'parameter_sensitivity' in self.visualization_data:
            for param, data in self.visualization_data['parameter_sensitivity'].items():
                optimal_params[f'optimal_{param}'] = data['optimal_value']
        
        return optimal_params
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成状态识别算法参数"""
        return {
            'algorithm_type': 'status_recognition',
            'supported_models': ['random_forest', 'svm', 'logistic_regression', 'decision_tree', 'gradient_boosting', 'knn'],
            'default_model': 'random_forest',
            'model_parameters': {
                'random_forest': {
                    'n_estimators': config.get('n_estimators', 100),
                    'max_depth': config.get('max_depth', None),
                    'min_samples_split': config.get('min_samples_split', 2),
                    'min_samples_leaf': config.get('min_samples_leaf', 1)
                },
                'svm': {
                    'C': config.get('C', 1.0),
                    'kernel': config.get('kernel', 'rbf'),
                    'gamma': config.get('gamma', 'scale')
                },
                'logistic_regression': {
                    'C': config.get('C', 1.0),
                    'penalty': config.get('penalty', 'l2'),
                    'solver': config.get('solver', 'lbfgs')
                }
            },
            'interactive_features': {
                'feature_selection': True,
                'threshold_adjustment': True,
                'parameter_tuning': True,
                'data_filtering': True,
                'outlier_handling': True,
                'data_sampling': True,
                'feature_transformation': True
            },
            'visualization_types': ['roc_curve', 'feature_importance', 'learning_curve', 'confusion_matrix', 'parameter_sensitivity'],
            'debug_features': ['data_distribution', 'outlier_analysis', 'correlation_analysis', 'data_quality', 'sampling_options']
        }


class HealthAssessmentTrainer(BaseTrainer):
    """健康度评估算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练健康度评估模型"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            train_data = self._load_data(config.train_data_path)
            
            # 准备特征
            X = train_data[config.feature_columns]
            
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 创建异常检测模型
            self.model = IsolationForest(
                contamination=config.algorithm_params.get('contamination', 0.1),
                random_state=42
            )
            
            # 训练模型
            self.model.fit(X_scaled)
            
            # 计算健康度分数
            health_scores = self.model.decision_function(X_scaled)
            
            # 计算评估指标
            # 对于异常检测，我们计算异常检测的准确率
            predictions = self.model.predict(X_scaled)
            # 假设大部分数据是正常的
            normal_ratio = np.sum(predictions == 1) / len(predictions)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 保存模型
            if config.save_model:
                self._save_model(self.model, config.output_path, config.model_format.value)
            
            return TrainingResult(
                task_id=f"health_assessment_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.HEALTH_ASSESSMENT,
                status=TrainingStatus.COMPLETED,
                accuracy=normal_ratio,  # 使用正常数据比例作为准确率
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/model.joblib",
                parameters_path=f"{config.output_path}/parameters.json",
                metadata={
                    'feature_columns': config.feature_columns,
                    'contamination': config.algorithm_params.get('contamination', 0.1),
                    'n_samples': len(train_data),
                    'health_score_range': [health_scores.min(), health_scores.max()]
                }
            )
            
        except Exception as e:
            logger.error(f"健康度评估模型训练失败: {str(e)}")
            raise
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成健康度评估算法参数"""
        return {
            'algorithm_type': 'health_assessment',
            'model_type': 'isolation_forest',
            'parameters': {
                'contamination': config.get('contamination', 0.1),
                'n_estimators': config.get('n_estimators', 100),
                'max_samples': config.get('max_samples', 'auto')
            },
            'thresholds': {
                'healthy_threshold': config.get('healthy_threshold', 0.0),
                'warning_threshold': config.get('warning_threshold', -0.5),
                'critical_threshold': config.get('critical_threshold', -1.0)
            }
        }


class VibrationAnalysisTrainer(BaseTrainer):
    """振动分析算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """生成振动分析参数（基于物理模型）"""
        start_time = datetime.now()
        
        try:
            # 振动算法不需要传统训练，而是基于物理参数计算
            bearing_params = config.algorithm_params.get('bearing_parameters', {})
            scenario_params = config.algorithm_params.get('scenario_parameters', {})
            
            # 计算振动特征参数
            vibration_params = self._calculate_vibration_parameters(bearing_params, scenario_params)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 保存参数
            if config.save_parameters:
                with open(f"{config.output_path}/vibration_parameters.json", 'w') as f:
                    json.dump(vibration_params, f, indent=2)
            
            return TrainingResult(
                task_id=f"vibration_analysis_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.VIBRATION_ANALYSIS,
                status=TrainingStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                parameters_path=f"{config.output_path}/vibration_parameters.json",
                metadata={
                    'bearing_type': bearing_params.get('bearing_type'),
                    'scenario_name': scenario_params.get('scenario_name'),
                    'parameter_count': len(vibration_params)
                }
            )
            
        except Exception as e:
            logger.error(f"振动分析参数生成失败: {str(e)}")
            raise
    
    def _calculate_vibration_parameters(self, bearing_params: Dict, scenario_params: Dict) -> Dict[str, Any]:
        """计算振动特征参数"""
        # 轴承参数
        inner_diameter = bearing_params.get('inner_diameter', 25)
        outer_diameter = bearing_params.get('outer_diameter', 52)
        ball_diameter = bearing_params.get('ball_diameter', 7.94)
        ball_count = bearing_params.get('ball_count', 9)
        
        # 计算特征频率
        shaft_speed = scenario_params.get('shaft_speed', 1500)  # RPM
        
        # 内圈故障频率
        fi = (shaft_speed / 60) * (ball_count / 2) * (1 + ball_diameter * np.cos(0) / inner_diameter)
        
        # 外圈故障频率
        fo = (shaft_speed / 60) * (ball_count / 2) * (1 - ball_diameter * np.cos(0) / outer_diameter)
        
        # 滚动体故障频率
        fb = (shaft_speed / 60) * (inner_diameter / ball_diameter) * (1 - (ball_diameter * np.cos(0) / inner_diameter) ** 2)
        
        # 保持架故障频率
        fc = (shaft_speed / 60) * (1 / 2) * (1 - ball_diameter * np.cos(0) / inner_diameter)
        
        return {
            'bearing_parameters': bearing_params,
            'scenario_parameters': scenario_params,
            'characteristic_frequencies': {
                'inner_race_frequency': round(fi, 2),
                'outer_race_frequency': round(fo, 2),
                'ball_frequency': round(fb, 2),
                'cage_frequency': round(fc, 2)
            },
            'analysis_parameters': {
                'sampling_frequency': scenario_params.get('sampling_frequency', 10000),
                'fft_size': scenario_params.get('fft_size', 2048),
                'window_type': scenario_params.get('window_type', 'hanning'),
                'overlap': scenario_params.get('overlap', 0.5)
            }
        }
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成振动分析算法参数"""
        return {
            'algorithm_type': 'vibration_analysis',
            'model_type': 'physical_model',
            'bearing_parameters': {
                'bearing_type': config.get('bearing_type', '6205'),
                'inner_diameter': config.get('inner_diameter', 25),
                'outer_diameter': config.get('outer_diameter', 52),
                'ball_diameter': config.get('ball_diameter', 7.94),
                'ball_count': config.get('ball_count', 9)
            },
            'scenario_parameters': {
                'scenario_name': config.get('scenario_name', 'normal_operation'),
                'shaft_speed': config.get('shaft_speed', 1500),
                'sampling_frequency': config.get('sampling_frequency', 10000),
                'fft_size': config.get('fft_size', 2048)
            }
        }


class AlarmTrainer(BaseTrainer):
    """报警算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练报警模型"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            train_data = self._load_data(config.train_data_path)
            
            # 准备特征和目标
            X = train_data[config.feature_columns]
            y = train_data[config.target_column]
            
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和验证数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 创建报警分类模型
            self.model = RandomForestClassifier(
                n_estimators=config.algorithm_params.get('n_estimators', 100),
                random_state=42
            )
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = self.model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # 保存模型
            if config.save_model:
                self._save_model(self.model, config.output_path, config.model_format.value)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TrainingResult(
                task_id=f"alarm_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.ALARM,
                status=TrainingStatus.COMPLETED,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/model.joblib",
                parameters_path=f"{config.output_path}/parameters.json",
                metadata={
                    'feature_columns': config.feature_columns,
                    'target_column': config.target_column,
                    'n_samples': len(train_data),
                    'alarm_levels': config.algorithm_params.get('alarm_levels', ['normal', 'warning', 'critical'])
                }
            )
            
        except Exception as e:
            logger.error(f"报警模型训练失败: {str(e)}")
            raise
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成报警算法参数"""
        return {
            'algorithm_type': 'alarm',
            'model_type': 'random_forest',
            'parameters': {
                'n_estimators': config.get('n_estimators', 100),
                'max_depth': config.get('max_depth', None),
                'min_samples_split': config.get('min_samples_split', 2)
            },
            'thresholds': {
                'warning_threshold': config.get('warning_threshold', 0.3),
                'critical_threshold': config.get('critical_threshold', 0.7)
            },
            'alarm_levels': config.get('alarm_levels', ['normal', 'warning', 'critical'])
        }


class SimulationTrainer(BaseTrainer):
    """仿真算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """生成仿真参数"""
        start_time = datetime.now()
        
        try:
            # 仿真算法不需要传统训练，而是基于配置生成仿真参数
            simulation_params = self._generate_simulation_parameters(config.algorithm_params)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 保存参数
            if config.save_parameters:
                with open(f"{config.output_path}/simulation_parameters.json", 'w') as f:
                    json.dump(simulation_params, f, indent=2)
            
            return TrainingResult(
                task_id=f"simulation_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.SIMULATION,
                status=TrainingStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                parameters_path=f"{config.output_path}/simulation_parameters.json",
                metadata={
                    'simulation_type': config.algorithm_params.get('simulation_type'),
                    'parameter_count': len(simulation_params)
                }
            )
            
        except Exception as e:
            logger.error(f"仿真参数生成失败: {str(e)}")
            raise
    
    def _generate_simulation_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真参数"""
        simulation_type = config.get('simulation_type', 'bearing_fault')
        
        if simulation_type == 'bearing_fault':
            return {
                'simulation_type': 'bearing_fault',
                'fault_types': ['inner_race', 'outer_race', 'ball', 'cage'],
                'fault_severity': config.get('fault_severity', [0.1, 0.3, 0.5, 0.7, 0.9]),
                'operating_conditions': {
                    'speed_range': config.get('speed_range', [500, 3000]),
                    'load_range': config.get('load_range', [0, 100]),
                    'temperature_range': config.get('temperature_range', [20, 80])
                },
                'signal_parameters': {
                    'sampling_frequency': config.get('sampling_frequency', 10000),
                    'duration': config.get('duration', 10),
                    'noise_level': config.get('noise_level', 0.05)
                }
            }
        elif simulation_type == 'gear_fault':
            return {
                'simulation_type': 'gear_fault',
                'fault_types': ['tooth_break', 'tooth_wear', 'eccentricity'],
                'gear_parameters': {
                    'teeth_count': config.get('teeth_count', 20),
                    'pressure_angle': config.get('pressure_angle', 20),
                    'module': config.get('module', 2)
                },
                'operating_conditions': {
                    'speed_range': config.get('speed_range', [500, 3000]),
                    'load_range': config.get('load_range', [0, 100])
                }
            }
        else:
            return {
                'simulation_type': simulation_type,
                'parameters': config
            }
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成仿真算法参数"""
        return {
            'algorithm_type': 'simulation',
            'model_type': 'parameter_based',
            'simulation_type': config.get('simulation_type', 'bearing_fault'),
            'parameters': config
        }


class TraditionalMLTrainer(BaseTrainer):
    """传统机器学习算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练传统机器学习模型"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            train_data = self._load_data(config.train_data_path)
            
            # 准备特征和目标
            X = train_data[config.feature_columns]
            y = train_data[config.target_column]
            
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和验证数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 根据算法类型创建模型
            model_type = config.algorithm_params.get('model_type', 'random_forest')
            self.model = self._create_model(model_type, config.algorithm_params)
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = self.model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # 保存模型
            if config.save_model:
                self._save_model(self.model, config.output_path, config.model_format.value)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TrainingResult(
                task_id=f"traditional_ml_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.TRADITIONAL_ML,
                status=TrainingStatus.COMPLETED,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/model.joblib",
                parameters_path=f"{config.output_path}/parameters.json",
                metadata={
                    'feature_columns': config.feature_columns,
                    'target_column': config.target_column,
                    'n_samples': len(train_data),
                    'model_type': model_type
                }
            )
            
        except Exception as e:
            logger.error(f"传统机器学习模型训练失败: {str(e)}")
            raise
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """创建指定类型的模型"""
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'svm':
            from sklearn.svm import SVC
            return SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                random_state=42
            )
        elif model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=params.get('C', 1.0),
                random_state=42
            )
        elif model_type == 'decision_tree':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=params.get('max_depth', None),
                random_state=42
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成传统机器学习算法参数"""
        model_type = config.get('model_type', 'random_forest')
        
        base_params = {
            'algorithm_type': 'traditional_ml',
            'model_type': model_type,
            'preprocessing': {
                'scaler': 'standard_scaler',
                'feature_selection': config.get('feature_selection', True)
            }
        }
        
        if model_type == 'random_forest':
            base_params['parameters'] = {
                'n_estimators': config.get('n_estimators', 100),
                'max_depth': config.get('max_depth', None),
                'min_samples_split': config.get('min_samples_split', 2)
            }
        elif model_type == 'svm':
            base_params['parameters'] = {
                'C': config.get('C', 1.0),
                'kernel': config.get('kernel', 'rbf'),
                'gamma': config.get('gamma', 'scale')
            }
        elif model_type == 'logistic_regression':
            base_params['parameters'] = {
                'C': config.get('C', 1.0),
                'penalty': config.get('penalty', 'l2'),
                'solver': config.get('solver', 'lbfgs')
            }
        
        return base_params


class DeepLearningTrainer(BaseTrainer):
    """深度学习算法训练器"""
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练深度学习模型"""
        start_time = datetime.now()
        
        try:
            # 加载数据
            train_data = self._load_data(config.train_data_path)
            
            # 准备特征和目标
            X = train_data[config.feature_columns]
            y = train_data[config.target_column]
            
            # 数据预处理
            X_scaled = self.scaler.fit_transform(X)
            
            # 分割训练和验证数据
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # 创建深度学习模型
            model_type = config.algorithm_params.get('model_type', 'mlp')
            self.model = self._create_deep_model(model_type, config.algorithm_params, X_train.shape[1])
            
            # 训练模型
            epochs = config.algorithm_params.get('epochs', 100)
            batch_size = config.algorithm_params.get('batch_size', 32)
            
            # 这里应该实现实际的深度学习训练逻辑
            # 由于需要TensorFlow/PyTorch，这里提供框架代码
            training_history = self._train_deep_model(
                self.model, X_train, y_train, X_val, y_val, epochs, batch_size
            )
            
            # 预测和评估
            y_pred = self.model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            # 保存模型
            if config.save_model:
                self._save_deep_model(self.model, config.output_path, config.model_format.value)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TrainingResult(
                task_id=f"deep_learning_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.DEEP_LEARNING,
                status=TrainingStatus.COMPLETED,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/model.h5",
                parameters_path=f"{config.output_path}/parameters.json",
                metadata={
                    'feature_columns': config.feature_columns,
                    'target_column': config.target_column,
                    'n_samples': len(train_data),
                    'model_type': model_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'training_history': training_history
                }
            )
            
        except Exception as e:
            logger.error(f"深度学习模型训练失败: {str(e)}")
            raise
    
    def _create_deep_model(self, model_type: str, params: Dict[str, Any], input_dim: int):
        """创建深度学习模型"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            if model_type == 'mlp':
                model = keras.Sequential([
                    keras.layers.Dense(params.get('hidden_units', [64, 32]), activation='relu', input_shape=(input_dim,)),
                    keras.layers.Dropout(params.get('dropout_rate', 0.2)),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return model
            else:
                raise ValueError(f"不支持的深度学习模型类型: {model_type}")
                
        except ImportError:
            logger.warning("TensorFlow未安装，无法创建深度学习模型")
            # 返回一个简单的sklearn模型作为替代
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _train_deep_model(self, model, X_train, y_train, X_val, y_val, epochs, batch_size):
        """训练深度学习模型"""
        try:
            # 实际的深度学习训练逻辑
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            return history.history
        except Exception as e:
            logger.warning(f"深度学习训练失败，使用替代方法: {str(e)}")
            # 如果深度学习训练失败，使用传统方法
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            return {}
    
    def _save_deep_model(self, model, output_path: str, model_format: str):
        """保存深度学习模型"""
        try:
            if model_format == 'h5':
                model.save(f"{output_path}/model.h5")
            elif model_format == 'json':
                # 保存模型参数
                model_params = self._extract_deep_model_params(model)
                with open(f"{output_path}/parameters.json", 'w') as f:
                    json.dump(model_params, f, indent=2)
        except Exception as e:
            logger.warning(f"深度学习模型保存失败: {str(e)}")
            # 使用传统方法保存
            self._save_model(model, output_path, 'pickle')
    
    def _extract_deep_model_params(self, model) -> Dict[str, Any]:
        """提取深度学习模型参数"""
        try:
            return {
                'model_type': type(model).__name__,
                'layers': [layer.get_config() for layer in model.layers],
                'optimizer': model.optimizer.get_config(),
                'loss': model.loss
            }
        except:
            return {'model_type': type(model).__name__}
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成深度学习算法参数"""
        return {
            'algorithm_type': 'deep_learning',
            'model_type': config.get('model_type', 'mlp'),
            'parameters': {
                'hidden_units': config.get('hidden_units', [64, 32]),
                'dropout_rate': config.get('dropout_rate', 0.2),
                'learning_rate': config.get('learning_rate', 0.001),
                'epochs': config.get('epochs', 100),
                'batch_size': config.get('batch_size', 32)
            },
            'preprocessing': {
                'scaler': 'standard_scaler',
                'feature_selection': config.get('feature_selection', True)
            }
        }


class VibrationTrainer(BaseTrainer):
    """振动算法训练器 - 支持交互式训练，无需数据预处理"""
    
    def __init__(self):
        super().__init__()
        self.current_model = None
        self.training_history = []
        self.visualization_data = {}
        self.interactive_params = {}
        self.raw_vibration_data = None
        self.processed_vibration_data = None
        self.vibration_analysis = {}
        self.real_time_config = {}
    
    async def train(self, config: TrainingConfig, data: Dict[str, Any]) -> TrainingResult:
        """训练振动分析模型 - 支持交互式训练"""
        start_time = datetime.now()
        
        try:
            # 加载振动数据
            vibration_data = self._load_vibration_data(config.train_data_path)
            self.raw_vibration_data = vibration_data.copy()
            
            # 解析振动算法配置
            algorithm_config = config.algorithm_params.get('vibration_config', {})
            
            # 应用交互式参数
            if self.interactive_params:
                vibration_data = self._apply_vibration_interactive_params(vibration_data, self.interactive_params)
            
            # 提取振动特征
            features = self._extract_vibration_features(vibration_data, algorithm_config)
            
            # 准备训练数据
            X = features
            y = vibration_data[config.target_column] if config.target_column in vibration_data.columns else None
            
            # 如果没有目标变量，进行无监督学习
            if y is None:
                # 使用异常检测模型
                from sklearn.ensemble import IsolationForest
                self.current_model = IsolationForest(
                    contamination=algorithm_config.get('contamination', 0.1),
                    random_state=42
                )
                self.current_model.fit(X)
                y_pred = self.current_model.predict(X)
                # 将异常检测结果转换为标签
                y = (y_pred == -1).astype(int)  # -1表示异常，1表示正常
            else:
                # 有监督学习
                model_type = algorithm_config.get('model_type', 'isolation_forest')
                self.current_model = self._create_vibration_model(model_type, algorithm_config)
                self.current_model.fit(X, y)
            
            # 生成振动分析结果
            self.vibration_analysis = self._analyze_vibration_data(vibration_data, features, algorithm_config)
            
            # 生成可视化数据
            self.visualization_data = self._generate_vibration_visualization(
                vibration_data, features, algorithm_config
            )
            
            # 计算性能指标
            if y is not None:
                y_pred = self.current_model.predict(X) if hasattr(self.current_model, 'predict') else y
                metrics = self._calculate_vibration_metrics(y, y_pred, features)
            else:
                metrics = {'anomaly_ratio': 0.1, 'feature_importance': {}}
            
            # 保存模型
            if config.save_model:
                self._save_model(self.current_model, config.output_path, config.model_format.value)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TrainingResult(
                task_id=f"vibration_{start_time.strftime('%Y%m%d_%H%M%S')}",
                algorithm_type=AlgorithmType.VIBRATION,
                status=TrainingStatus.COMPLETED,
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                model_path=f"{config.output_path}/vibration_model.joblib",
                parameters_path=f"{config.output_path}/vibration_parameters.json",
                metadata={
                    'vibration_config': algorithm_config,
                    'interactive_params': self.interactive_params,
                    'vibration_analysis': self.vibration_analysis,
                    'visualization_data': self.visualization_data,
                    'feature_columns': list(features.columns) if hasattr(features, 'columns') else [],
                    'target_column': config.target_column,
                    'n_samples': len(vibration_data),
                    'sampling_rate': algorithm_config.get('sampling_rate', 1000),
                    'frequency_range': algorithm_config.get('frequency_range', [0, 1000])
                }
            )
            
        except Exception as e:
            logger.error(f"振动模型训练失败: {str(e)}")
            raise
    
    def _load_vibration_data(self, data_path: str) -> pd.DataFrame:
        """加载振动数据"""
        try:
            # 支持多种振动数据格式
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            elif data_path.endswith('.h5'):
                data = pd.read_hdf(data_path)
            else:
                # 假设是二进制振动数据
                data = self._load_binary_vibration_data(data_path)
            
            return data
        except Exception as e:
            logger.error(f"加载振动数据失败: {str(e)}")
            raise
    
    def _load_binary_vibration_data(self, data_path: str) -> pd.DataFrame:
        """加载二进制振动数据"""
        # 这里实现二进制振动数据的加载逻辑
        # 根据实际数据格式进行解析
        import numpy as np
        
        # 示例：假设数据格式为 [timestamp, x_accel, y_accel, z_accel, speed]
        data = np.fromfile(data_path, dtype=np.float32)
        n_samples = len(data) // 5  # 假设每行5个值
        
        data = data.reshape(n_samples, 5)
        df = pd.DataFrame(data, columns=['timestamp', 'x_accel', 'y_accel', 'z_accel', 'speed'])
        
        return df
    
    def _apply_vibration_interactive_params(self, data: pd.DataFrame, interactive_params: Dict[str, Any]) -> pd.DataFrame:
        """应用振动算法的交互式参数"""
        processed_data = data.copy()
        
        # 1. 频率过滤
        if 'frequency_filtering' in interactive_params:
            freq_config = interactive_params['frequency_filtering']
            if freq_config.get('enabled', False):
                processed_data = self._apply_frequency_filtering(processed_data, freq_config)
        
        # 2. 振幅阈值过滤
        if 'amplitude_thresholds' in interactive_params:
            threshold_config = interactive_params['amplitude_thresholds']
            processed_data = self._apply_amplitude_thresholds(processed_data, threshold_config)
        
        # 3. 数据选择
        if 'data_selection' in interactive_params:
            selection_config = interactive_params['data_selection']
            processed_data = self._apply_data_selection(processed_data, selection_config)
        
        # 4. 实时调整参数
        if 'real_time_adjustment' in interactive_params:
            adjustment_config = interactive_params['real_time_adjustment']
            self.real_time_config.update(adjustment_config)
        
        return processed_data
    
    def _apply_frequency_filtering(self, data: pd.DataFrame, freq_config: Dict[str, Any]) -> pd.DataFrame:
        """应用频率过滤"""
        from scipy import signal
        
        # 获取振动信号列
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal_data = data[col].values
            
            # 低通滤波
            if 'low_freq_cutoff' in freq_config:
                low_cutoff = freq_config['low_freq_cutoff']
                sampling_rate = freq_config.get('sampling_rate', 1000)
                nyquist = sampling_rate / 2
                low_cutoff_norm = low_cutoff / nyquist
                b, a = signal.butter(4, low_cutoff_norm, btype='low')
                signal_data = signal.filtfilt(b, a, signal_data)
            
            # 高通滤波
            if 'high_freq_cutoff' in freq_config:
                high_cutoff = freq_config['high_freq_cutoff']
                sampling_rate = freq_config.get('sampling_rate', 1000)
                nyquist = sampling_rate / 2
                high_cutoff_norm = high_cutoff / nyquist
                b, a = signal.butter(4, high_cutoff_norm, btype='high')
                signal_data = signal.filtfilt(b, a, signal_data)
            
            # 带通滤波
            if 'bandpass_filters' in freq_config:
                for band_filter in freq_config['bandpass_filters']:
                    center_freq = band_filter['center']
                    bandwidth = band_filter['bandwidth']
                    sampling_rate = freq_config.get('sampling_rate', 1000)
                    nyquist = sampling_rate / 2
                    
                    low_freq = (center_freq - bandwidth/2) / nyquist
                    high_freq = (center_freq + bandwidth/2) / nyquist
                    
                    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
                    signal_data = signal.filtfilt(b, a, signal_data)
            
            data[col] = signal_data
        
        return data
    
    def _apply_amplitude_thresholds(self, data: pd.DataFrame, threshold_config: Dict[str, Any]) -> pd.DataFrame:
        """应用振幅阈值过滤"""
        # 获取振动信号列
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal_data = data[col].values
            
            # 计算RMS值
            rms = np.sqrt(np.mean(signal_data**2))
            
            # 应用阈值
            warning_level = threshold_config.get('warning_level', 0.5)
            alarm_level = threshold_config.get('alarm_level', 1.0)
            critical_level = threshold_config.get('critical_level', 2.0)
            
            # 标记异常数据
            if rms > critical_level:
                data[f'{col}_status'] = 'critical'
            elif rms > alarm_level:
                data[f'{col}_status'] = 'alarm'
            elif rms > warning_level:
                data[f'{col}_status'] = 'warning'
            else:
                data[f'{col}_status'] = 'normal'
        
        return data
    
    def _apply_data_selection(self, data: pd.DataFrame, selection_config: Dict[str, Any]) -> pd.DataFrame:
        """应用数据选择"""
        # 时间范围选择
        if 'time_range' in selection_config:
            time_config = selection_config['time_range']
            if 'timestamp' in data.columns:
                start_time = pd.to_datetime(time_config.get('start_time'))
                end_time = pd.to_datetime(time_config.get('end_time'))
                data = data[(data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)]
        
        # 转速范围选择
        if 'speed_range' in selection_config:
            speed_config = selection_config['speed_range']
            if 'speed' in data.columns:
                min_speed = speed_config.get('min_speed', 0)
                max_speed = speed_config.get('max_speed', float('inf'))
                data = data[(data['speed'] >= min_speed) & (data['speed'] <= max_speed)]
        
        # 质量过滤
        if 'quality_filters' in selection_config:
            quality_config = selection_config['quality_filters']
            if 'signal_quality' in data.columns:
                min_quality = quality_config.get('min_signal_quality', 0)
                data = data[data['signal_quality'] >= min_quality]
        
        return data
    
    def _extract_vibration_features(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """提取振动特征"""
        features = {}
        
        # 获取振动信号列
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal_data = data[col].values
            
            # 时域特征
            if config.get('time_domain', {}).get('rms_enabled', True):
                features[f'{col}_rms'] = np.sqrt(np.mean(signal_data**2))
            
            if config.get('time_domain', {}).get('peak_enabled', True):
                features[f'{col}_peak'] = np.max(np.abs(signal_data))
            
            if config.get('time_domain', {}).get('crest_factor_enabled', True):
                rms = np.sqrt(np.mean(signal_data**2))
                peak = np.max(np.abs(signal_data))
                features[f'{col}_crest_factor'] = peak / rms if rms > 0 else 0
            
            if config.get('time_domain', {}).get('kurtosis_enabled', True):
                features[f'{col}_kurtosis'] = self._calculate_kurtosis(signal_data)
            
            # 频域特征
            if config.get('frequency_domain', {}).get('spectrum_enabled', True):
                freq_features = self._extract_frequency_features(signal_data, config)
                features.update({f'{col}_{k}': v for k, v in freq_features.items()})
        
        # 添加转速相关特征
        if 'speed' in data.columns:
            features['speed_mean'] = data['speed'].mean()
            features['speed_std'] = data['speed'].std()
            features['speed_range'] = data['speed'].max() - data['speed'].min()
        
        return pd.DataFrame([features])
    
    def _extract_frequency_features(self, signal_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        """提取频域特征"""
        from scipy import signal
        
        features = {}
        sampling_rate = config.get('sampling_rate', 1000)
        
        # FFT分析
        fft_result = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(len(signal_data), 1/sampling_rate)
        
        # 功率谱密度
        psd = np.abs(fft_result)**2
        
        # 主要频率成分
        dominant_freq_idx = np.argmax(psd[:len(psd)//2])
        features['dominant_frequency'] = frequencies[dominant_freq_idx]
        features['dominant_amplitude'] = psd[dominant_freq_idx]
        
        # 频带能量
        freq_ranges = config.get('frequency_ranges', [
            (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000)
        ])
        
        for i, (low_freq, high_freq) in enumerate(freq_ranges):
            mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            band_energy = np.sum(psd[mask])
            features[f'band_{i}_energy'] = band_energy
        
        return features
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峭度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _analyze_vibration_data(self, data: pd.DataFrame, features: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """分析振动数据"""
        analysis = {}
        
        # 1. 信号质量分析
        analysis['signal_quality'] = self._analyze_signal_quality(data)
        
        # 2. 频率分析
        analysis['frequency_analysis'] = self._analyze_frequency_content(data, config)
        
        # 3. 趋势分析
        analysis['trend_analysis'] = self._analyze_trends(data)
        
        # 4. 异常检测
        analysis['anomaly_detection'] = self._detect_vibration_anomalies(data, features)
        
        return analysis
    
    def _analyze_signal_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析信号质量"""
        quality = {}
        
        # 计算信噪比
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        for col in vibration_cols:
            signal = data[col].values
            
            # 简单的信噪比估计
            signal_power = np.mean(signal**2)
            noise_power = np.var(signal - np.mean(signal))
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            quality[f'{col}_snr'] = snr
            quality[f'{col}_signal_strength'] = np.sqrt(signal_power)
        
        return quality
    
    def _analyze_frequency_content(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """分析频率内容"""
        freq_analysis = {}
        
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        sampling_rate = config.get('sampling_rate', 1000)
        
        for col in vibration_cols:
            signal = data[col].values
            
            # FFT分析
            fft_result = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
            psd = np.abs(fft_result)**2
            
            # 主要频率成分
            freq_analysis[f'{col}_main_frequencies'] = frequencies[np.argsort(psd)[-5:]].tolist()
            freq_analysis[f'{col}_frequency_peaks'] = psd[np.argsort(psd)[-5:]].tolist()
        
        return freq_analysis
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析趋势"""
        trends = {}
        
        # 分析时间序列趋势
        if 'timestamp' in data.columns:
            data_sorted = data.sort_values('timestamp')
            
            for col in data.columns:
                if col != 'timestamp' and data[col].dtype in ['float64', 'int64']:
                    # 简单的线性趋势分析
                    x = np.arange(len(data_sorted))
                    y = data_sorted[col].values
                    
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        trends[f'{col}_trend_slope'] = slope
                        trends[f'{col}_trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
        
        return trends
    
    def _detect_vibration_anomalies(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """检测振动异常"""
        anomalies = {}
        
        # 基于统计的异常检测
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                values = features[col].values
                mean = np.mean(values)
                std = np.std(values)
                
                # 3-sigma规则
                threshold_high = mean + 3 * std
                threshold_low = mean - 3 * std
                
                anomalies[f'{col}_anomaly_threshold_high'] = threshold_high
                anomalies[f'{col}_anomaly_threshold_low'] = threshold_low
                anomalies[f'{col}_anomaly_count'] = np.sum((values > threshold_high) | (values < threshold_low))
        
        return anomalies
    
    def _generate_vibration_visualization(self, data: pd.DataFrame, features: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成振动可视化数据"""
        viz_data = {}
        
        # 1. 时域波形图
        vibration_cols = [col for col in data.columns if 'accel' in col.lower() or 'vibration' in col.lower()]
        
        viz_data['time_domain_waveforms'] = {}
        for col in vibration_cols:
            # 取前1000个点用于显示
            sample_size = min(1000, len(data))
            viz_data['time_domain_waveforms'][col] = {
                'time': np.arange(sample_size).tolist(),
                'amplitude': data[col].iloc[:sample_size].tolist()
            }
        
        # 2. 频谱图
        viz_data['frequency_spectrums'] = {}
        sampling_rate = config.get('sampling_rate', 1000)
        
        for col in vibration_cols:
            signal = data[col].values
            fft_result = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
            psd = np.abs(fft_result)**2
            
            # 只取正频率部分
            positive_freq_mask = frequencies >= 0
            viz_data['frequency_spectrums'][col] = {
                'frequencies': frequencies[positive_freq_mask].tolist(),
                'power_spectrum': psd[positive_freq_mask].tolist()
            }
        
        # 3. 特征分布图
        viz_data['feature_distributions'] = {}
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                values = features[col].values
                viz_data['feature_distributions'][col] = {
                    'values': values.tolist(),
                    'histogram': np.histogram(values, bins=20)[0].tolist(),
                    'bin_edges': np.histogram(values, bins=20)[1].tolist()
                }
        
        # 4. 趋势图
        if 'timestamp' in data.columns:
            viz_data['trends'] = {}
            for col in data.columns:
                if col != 'timestamp' and data[col].dtype in ['float64', 'int64']:
                    data_sorted = data.sort_values('timestamp')
                    viz_data['trends'][col] = {
                        'timestamps': data_sorted['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                        'values': data_sorted[col].tolist()
                    }
        
        return viz_data
    
    async def get_vibration_analysis(self) -> Dict[str, Any]:
        """获取振动分析结果"""
        return self.vibration_analysis
    
    async def get_vibration_visualization(self) -> Dict[str, Any]:
        """获取振动可视化数据"""
        return self.visualization_data
    
    async def get_real_time_config(self) -> Dict[str, Any]:
        """获取实时配置"""
        return self.real_time_config
    
    async def generate_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成振动算法参数"""
        return {
            'algorithm_type': 'vibration',
            'vibration_config': {
                'sampling_rate': config.get('sampling_rate', 1000),
                'data_type': 'float32',
                'model_type': 'isolation_forest',
                'contamination': 0.1,
                'frequency_range': [0, 1000],
                'time_domain': {
                    'rms_enabled': True,
                    'peak_enabled': True,
                    'crest_factor_enabled': True,
                    'kurtosis_enabled': True
                },
                'frequency_domain': {
                    'spectrum_enabled': True,
                    'harmonic_analysis': True,
                    'sideband_analysis': True,
                    'envelope_analysis': True
                }
            },
            'interactive_features': {
                'frequency_filtering': True,
                'amplitude_thresholds': True,
                'data_selection': True,
                'real_time_adjustment': True,
                'feature_weights': True
            },
            'visualization_types': [
                'time_domain_waveforms', 'frequency_spectrums', 
                'feature_distributions', 'trends', 'anomaly_detection'
            ],
            'real_time_features': [
                'adaptive_thresholds', 'dynamic_filtering', 
                'online_learning', 'performance_monitoring'
            ]
        }


class TrainerFactory:
    """训练器工厂类"""
    
    _trainers = {
        AlgorithmType.STATUS_RECOGNITION: StatusRecognitionTrainer,
        AlgorithmType.HEALTH_ASSESSMENT: HealthAssessmentTrainer,
        AlgorithmType.VIBRATION: VibrationTrainer,
        AlgorithmType.ALARM: AlarmTrainer,
        AlgorithmType.SIMULATION: SimulationTrainer,
        AlgorithmType.TRADITIONAL_ML: TraditionalMLTrainer,
        AlgorithmType.DEEP_LEARNING: DeepLearningTrainer
    }
    
    @classmethod
    def get_trainer(cls, algorithm_type: AlgorithmType) -> BaseTrainer:
        """获取指定算法类型的训练器"""
        if algorithm_type not in cls._trainers:
            raise ValueError(f"不支持的算法类型: {algorithm_type}")
        
        return cls._trainers[algorithm_type]()
    
    @classmethod
    def get_supported_algorithms(cls) -> list:
        """获取支持的算法类型列表"""
        return list(cls._trainers.keys())


# 异步训练管理器
class AsyncTrainingManager:
    """异步训练管理器"""
    
    def __init__(self):
        self.active_tasks = {}
        self.task_results = {}
    
    async def start_training(self, config: TrainingConfig, data: Dict[str, Any]) -> str:
        """启动异步训练任务"""
        task_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建训练任务
        trainer = TrainerFactory.get_trainer(config.algorithm_type)
        
        # 启动异步任务
        task = asyncio.create_task(self._run_training(task_id, trainer, config, data))
        self.active_tasks[task_id] = task
        
        logger.info(f"启动训练任务: {task_id}")
        return task_id
    
    async def _run_training(self, task_id: str, trainer: BaseTrainer, config: TrainingConfig, data: Dict[str, Any]):
        """运行训练任务"""
        try:
            result = await trainer.train(config, data)
            self.task_results[task_id] = result
            logger.info(f"训练任务完成: {task_id}")
        except Exception as e:
            logger.error(f"训练任务失败: {task_id}, 错误: {str(e)}")
            self.task_results[task_id] = TrainingResult(
                task_id=task_id,
                algorithm_type=config.algorithm_type,
                status=TrainingStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration=0,
                error_message=str(e)
            )
        finally:
            # 清理任务
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def get_task_status(self, task_id: str) -> Optional[TrainingResult]:
        """获取任务状态"""
        if task_id in self.active_tasks:
            # 任务正在运行
            return TrainingResult(
                task_id=task_id,
                algorithm_type=AlgorithmType.STATUS_RECOGNITION,  # 默认值
                status=TrainingStatus.RUNNING,
                start_time=datetime.now(),
                end_time=None,
                duration=0
            )
        elif task_id in self.task_results:
            # 任务已完成
            return self.task_results[task_id]
        else:
            # 任务不存在
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消训练任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.cancel()
            del self.active_tasks[task_id]
            logger.info(f"取消训练任务: {task_id}")
            return True
        return False
    
    async def get_all_tasks(self) -> Dict[str, TrainingResult]:
        """获取所有任务状态"""
        all_tasks = {}
        
        # 添加正在运行的任务
        for task_id in self.active_tasks:
            all_tasks[task_id] = TrainingResult(
                task_id=task_id,
                algorithm_type=AlgorithmType.STATUS_RECOGNITION,  # 默认值
                status=TrainingStatus.RUNNING,
                start_time=datetime.now(),
                end_time=None,
                duration=0
            )
        
        # 添加已完成的任务
        all_tasks.update(self.task_results)
        
        return all_tasks


# 参数生成器
class ParameterGenerator:
    """参数生成器"""
    
    @staticmethod
    async def generate_parameters(algorithm_type: AlgorithmType, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成算法参数"""
        trainer = TrainerFactory.get_trainer(algorithm_type)
        return await trainer.generate_parameters(config)
    
    @staticmethod
    def generate_random_parameters(algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """生成随机参数"""
        import random
        
        if algorithm_type == AlgorithmType.STATUS_RECOGNITION:
            return {
                'n_estimators': random.randint(50, 200),
                'max_depth': random.randint(5, 20) if random.random() > 0.5 else None,
                'min_samples_split': random.randint(2, 10),
                'min_samples_leaf': random.randint(1, 5)
            }
        elif algorithm_type == AlgorithmType.HEALTH_ASSESSMENT:
            return {
                'contamination': round(random.uniform(0.05, 0.2), 3),
                'n_estimators': random.randint(50, 200),
                'max_samples': random.choice(['auto', 'sqrt', 'log2'])
            }
        elif algorithm_type == AlgorithmType.DEEP_LEARNING:
            return {
                'hidden_units': [random.randint(32, 128), random.randint(16, 64)],
                'dropout_rate': round(random.uniform(0.1, 0.5), 2),
                'learning_rate': round(random.uniform(0.0001, 0.01), 4),
                'epochs': random.randint(50, 200),
                'batch_size': random.choice([16, 32, 64, 128])
            }
        else:
            return {}


# 模型评估器
class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
        """评估模型性能"""
        try:
            y_pred = model.predict(X_test)
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    @staticmethod
    def compare_models(models: Dict[str, Any], X_test, y_test) -> Dict[str, Dict[str, float]]:
        """比较多个模型性能"""
        results = {}
        for model_name, model in models.items():
            results[model_name] = ModelEvaluator.evaluate_model(model, X_test, y_test)
        return results


# 模型保存和加载器
class ModelManager:
    """模型管理器"""
    
    @staticmethod
    def save_model(model, filepath: str, format: str = 'joblib'):
        """保存模型"""
        try:
            if format == 'joblib':
                joblib.dump(model, filepath)
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            elif format == 'json':
                # 对于sklearn模型，保存参数为JSON
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    with open(filepath, 'w') as f:
                        json.dump(params, f, indent=2)
            else:
                raise ValueError(f"不支持的模型格式: {format}")
            
            logger.info(f"模型保存成功: {filepath}")
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
    
    @staticmethod
    def load_model(filepath: str, format: str = 'joblib'):
        """加载模型"""
        try:
            if format == 'joblib':
                return joblib.load(filepath)
            elif format == 'pickle':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"不支持的模型格式: {format}")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise


# 数据预处理器
class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = ""
    
    def fit_transform(self, data: pd.DataFrame, feature_columns: list, target_column: str = None):
        """拟合并转换数据"""
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        X = data[feature_columns]
        X_scaled = self.scaler.fit_transform(X)
        
        if target_column:
            y = data[target_column]
            return X_scaled, y
        else:
            return X_scaled
    
    def transform(self, data: pd.DataFrame):
        """转换数据"""
        X = data[self.feature_columns]
        return self.scaler.transform(X)
    
    def inverse_transform(self, X_scaled):
        """逆转换数据"""
        return self.scaler.inverse_transform(X_scaled)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.feature_columns, model.feature_importances_))
        else:
            return {}


# 训练历史记录器
class TrainingHistoryLogger:
    """训练历史记录器"""
    
    def __init__(self):
        self.history = []
    
    def log_training_step(self, epoch: int, metrics: Dict[str, float]):
        """记录训练步骤"""
        self.history.append({
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
    
    def get_best_epoch(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """获取最佳epoch"""
        if not self.history:
            return {}
        
        best_step = max(self.history, key=lambda x: x['metrics'].get(metric, 0))
        return best_step
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.history:
            return {}
        
        return {
            'total_epochs': len(self.history),
            'best_accuracy': max([h['metrics'].get('accuracy', 0) for h in self.history]),
            'final_accuracy': self.history[-1]['metrics'].get('accuracy', 0),
            'training_duration': len(self.history)
        }
    
    def save_history(self, filepath: str):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, filepath: str):
        """加载训练历史"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)