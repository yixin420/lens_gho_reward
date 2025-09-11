"""
模型评估模块
提供详细的模型评估指标和分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.results = {}
        
    def evaluate(self, y_true: List[int], y_pred: List[int], 
                y_proba: Optional[np.ndarray] = None,
                class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        全面评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（可选）
            class_names: 类别名称
            
        Returns:
            评估指标字典
        """
        if class_names is None:
            class_names = ['负面', '正面']
        
        # 基础指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 如果提供了概率，计算AUC
        if y_proba is not None and len(np.unique(y_true)) == 2:
            # 二分类情况
            if y_proba.ndim == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
            metrics['auc'] = auc(fpr, tpr)
            metrics['average_precision'] = average_precision_score(y_true, y_proba_positive)
        
        # 保存结果
        self.results = metrics
        
        return metrics
    
    def print_report(self, y_true: List[int], y_pred: List[int], 
                    class_names: Optional[List[str]] = None):
        """
        打印详细的分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
        """
        if class_names is None:
            class_names = ['负面', '正面']
        
        print("\n" + "="*50)
        print("分类报告")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print("\n混淆矩阵:")
        print("-"*30)
        print(f"{'':10} {'预测负面':>10} {'预测正面':>10}")
        print(f"{'真实负面':10} {cm[0,0]:>10} {cm[0,1]:>10}")
        print(f"{'真实正面':10} {cm[1,0]:>10} {cm[1,1]:>10}")
        
        # 计算额外指标
        total = len(y_true)
        correct = np.sum(y_true == y_pred)
        incorrect = total - correct
        
        print("\n统计信息:")
        print("-"*30)
        print(f"总样本数: {total}")
        print(f"正确预测: {correct} ({correct/total*100:.2f}%)")
        print(f"错误预测: {incorrect} ({incorrect/total*100:.2f}%)")
        
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int],
                            class_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None):
        """
        绘制混淆矩阵热力图
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_path: 保存路径
        """
        if class_names is None:
            class_names = ['负面', '正面']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"混淆矩阵图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, y_true: List[int], y_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"ROC曲线图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: List[int], y_proba: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        绘制精确率-召回率曲线
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            save_path: 保存路径
        """
        if y_proba.ndim == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = y_proba
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        average_precision = average_precision_score(y_true, y_proba_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'平均精确率 = {average_precision:.2f}')
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc="lower left")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"精确率-召回率曲线图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_curve(self, model, X, y, cv=5, 
                          train_sizes=np.linspace(0.1, 1.0, 10),
                          save_path: Optional[str] = None):
        """
        绘制学习曲线
        
        Args:
            model: 模型
            X: 特征
            y: 标签
            cv: 交叉验证折数
            train_sizes: 训练集大小
            save_path: 保存路径
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
        plt.plot(train_sizes, val_mean, 'o-', color='green', label='验证分数')
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.1, color='blue')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='green')
        
        plt.xlabel('训练样本数')
        plt.ylabel('准确率')
        plt.title('学习曲线')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"学习曲线图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def error_analysis(self, texts: List[str], y_true: List[int], 
                       y_pred: List[int], num_examples: int = 10) -> Dict:
        """
        错误分析
        
        Args:
            texts: 原始文本
            y_true: 真实标签
            y_pred: 预测标签
            num_examples: 显示的错误样本数量
            
        Returns:
            错误分析结果
        """
        # 找出错误预测的索引
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'index': i,
                    'text': texts[i],
                    'true_label': y_true[i],
                    'pred_label': y_pred[i]
                })
        
        # 统计错误类型
        false_positives = sum(1 for e in errors if e['true_label'] == 0 and e['pred_label'] == 1)
        false_negatives = sum(1 for e in errors if e['true_label'] == 1 and e['pred_label'] == 0)
        
        analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'error_examples': errors[:num_examples]
        }
        
        # 打印错误分析
        print("\n" + "="*50)
        print("错误分析")
        print("="*50)
        print(f"总错误数: {analysis['total_errors']}")
        print(f"错误率: {analysis['error_rate']:.2%}")
        print(f"假阳性（负面误判为正面）: {false_positives}")
        print(f"假阴性（正面误判为负面）: {false_negatives}")
        
        print(f"\n前{num_examples}个错误样本:")
        print("-"*50)
        for i, error in enumerate(analysis['error_examples'], 1):
            label_map = {0: '负面', 1: '正面'}
            print(f"\n样本 {i}:")
            print(f"文本: {error['text'][:100]}...")
            print(f"真实标签: {label_map[error['true_label']]}")
            print(f"预测标签: {label_map[error['pred_label']]}")
        
        return analysis
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            results_dict: 模型结果字典 {model_name: metrics}
            
        Returns:
            比较结果DataFrame
        """
        
        comparison = []
        for model_name, metrics in results_dict.items():
            comparison.append({
                '模型': model_name,
                '准确率': metrics.get('accuracy', 0),
                '精确率': metrics.get('precision', 0),
                '召回率': metrics.get('recall', 0),
                'F1分数': metrics.get('f1_score', 0),
                'AUC': metrics.get('auc', 0)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1分数', ascending=False)
        
        print("\n" + "="*50)
        print("模型性能比较")
        print("="*50)
        print(df.to_string(index=False))
        
        return df
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict],
                            save_path: Optional[str] = None):
        """
        绘制模型比较图
        
        Args:
            results_dict: 模型结果字典
            save_path: 保存路径
        """
        models = list(results_dict.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
            values = [results_dict[model].get(metric_name, 0) for model in models]
            
            axes[idx].bar(models, values, color=['blue', 'green', 'red', 'orange', 'purple'][:len(models)])
            axes[idx].set_title(metric_label)
            axes[idx].set_ylim([0, 1])
            axes[idx].set_ylabel('分数')
            
            # 在柱状图上添加数值
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.suptitle('模型性能比较', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"模型比较图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()