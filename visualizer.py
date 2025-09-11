"""
可视化模块
提供情绪分析结果的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SentimentVisualizer:
    """情绪分析可视化器"""
    
    def __init__(self, language='zh'):
        """
        初始化可视化器
        
        Args:
            language: 语言类型
        """
        self.language = language
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        
        # 颜色方案
        self.colors = {
            'positive': '#2ecc71',  # 绿色
            'negative': '#e74c3c',  # 红色
            'neutral': '#95a5a6'    # 灰色
        }
    
    def plot_sentiment_distribution(self, sentiments: List[int], 
                                   labels: Optional[List[str]] = None,
                                   save_path: Optional[str] = None):
        """
        绘制情绪分布图
        
        Args:
            sentiments: 情绪标签列表
            labels: 标签名称
            save_path: 保存路径
        """
        if labels is None:
            labels = ['负面', '正面']
        
        # 统计各类别数量
        unique, counts = np.unique(sentiments, return_counts=True)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 柱状图
        colors = [self.colors['negative'], self.colors['positive']]
        bars = ax1.bar(labels, counts, color=colors[:len(labels)])
        ax1.set_title('情绪分布统计')
        ax1.set_ylabel('数量')
        
        # 在柱状图上添加数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}\n({count/sum(counts)*100:.1f}%)',
                    ha='center', va='bottom')
        
        # 饼图
        ax2.pie(counts, labels=labels, colors=colors[:len(labels)],
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('情绪比例分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"情绪分布图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_sentiment_timeline(self, timestamps: List, sentiments: List[int],
                               window_size: int = 10,
                               save_path: Optional[str] = None):
        """
        绘制情绪时间线
        
        Args:
            timestamps: 时间戳列表
            sentiments: 情绪标签列表
            window_size: 移动平均窗口大小
            save_path: 保存路径
        """
        # 计算移动平均
        sentiment_values = np.array(sentiments)
        moving_avg = np.convolve(sentiment_values, 
                                 np.ones(window_size)/window_size, 
                                 mode='valid')
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 原始情绪值
        ax1.scatter(range(len(sentiments)), sentiments, 
                   c=['red' if s == 0 else 'green' for s in sentiments],
                   alpha=0.6, s=20)
        ax1.set_title('情绪变化时间线')
        ax1.set_ylabel('情绪（0=负面, 1=正面）')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # 移动平均
        ax2.plot(range(len(moving_avg)), moving_avg, 
                color='blue', linewidth=2, label=f'移动平均(窗口={window_size})')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(moving_avg)), 0, moving_avg,
                        where=(moving_avg > 0.5), color='green', alpha=0.3, label='正面趋势')
        ax2.fill_between(range(len(moving_avg)), 0, moving_avg,
                        where=(moving_avg <= 0.5), color='red', alpha=0.3, label='负面趋势')
        ax2.set_title('情绪趋势（移动平均）')
        ax2.set_xlabel('样本序号')
        ax2.set_ylabel('平均情绪值')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"情绪时间线图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confidence_distribution(self, probabilities: np.ndarray,
                                    true_labels: Optional[List[int]] = None,
                                    save_path: Optional[str] = None):
        """
        绘制置信度分布图
        
        Args:
            probabilities: 预测概率
            true_labels: 真实标签（可选）
            save_path: 保存路径
        """
        # 获取正面情绪的概率
        if probabilities.ndim == 2:
            positive_probs = probabilities[:, 1]
        else:
            positive_probs = probabilities
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图
        axes[0].hist(positive_probs, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_title('预测置信度分布')
        axes[0].set_xlabel('正面情绪概率')
        axes[0].set_ylabel('频数')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='决策边界')
        axes[0].legend()
        
        # 如果提供了真实标签，绘制分类别的置信度分布
        if true_labels is not None:
            positive_correct = positive_probs[np.array(true_labels) == 1]
            negative_correct = positive_probs[np.array(true_labels) == 0]
            
            axes[1].hist(negative_correct, bins=15, alpha=0.5, 
                        label='真实负面', color='red', edgecolor='black')
            axes[1].hist(positive_correct, bins=15, alpha=0.5,
                        label='真实正面', color='green', edgecolor='black')
            axes[1].set_title('按真实标签的置信度分布')
            axes[1].set_xlabel('正面情绪概率')
            axes[1].set_ylabel('频数')
            axes[1].axvline(x=0.5, color='black', linestyle='--', label='决策边界')
            axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"置信度分布图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_wordcloud(self, texts: List[str], sentiments: List[int],
                        sentiment_filter: Optional[int] = None,
                        save_path: Optional[str] = None):
        """
        创建词云图
        
        Args:
            texts: 文本列表
            sentiments: 情绪标签列表
            sentiment_filter: 筛选特定情绪（0=负面，1=正面，None=全部）
            save_path: 保存路径
        """
        # 筛选文本
        if sentiment_filter is not None:
            filtered_texts = [text for text, sent in zip(texts, sentiments) 
                            if sent == sentiment_filter]
            title = f"{'正面' if sentiment_filter == 1 else '负面'}情绪词云"
        else:
            filtered_texts = texts
            title = "全部文本词云"
        
        # 合并文本
        combined_text = ' '.join(filtered_texts)
        
        # 创建词云
        if self.language == 'zh':
            # 中文词云需要指定字体
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                font_path='simhei.ttf' if self._check_font('simhei.ttf') else None,
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(combined_text)
        else:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(combined_text)
        
        # 绘制词云
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"词云图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: List[float],
                               top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称
            importances: 特征重要性
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        # 排序并选择前N个
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 水平条形图
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importances, color='steelblue')
        plt.yticks(y_pos, top_features)
        plt.xlabel('重要性分数')
        plt.title(f'前{top_n}个重要特征')
        plt.gca().invert_yaxis()  # 反转y轴，最重要的在顶部
        
        # 添加数值标签
        for i, v in enumerate(top_importances):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"特征重要性图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_sentiment_heatmap(self, texts: List[str], sentiments: List[int],
                              categories: Optional[List[str]] = None,
                              save_path: Optional[str] = None):
        """
        绘制情绪热力图
        
        Args:
            texts: 文本列表
            sentiments: 情绪标签列表
            categories: 类别列表（可选）
            save_path: 保存路径
        """
        if categories is None:
            # 如果没有提供类别，使用索引
            categories = [f"类别{i//10}" for i in range(len(sentiments))]
        
        # 创建数据矩阵
        unique_categories = list(set(categories))
        sentiment_matrix = np.zeros((len(unique_categories), 2))
        
        for cat, sent in zip(categories, sentiments):
            cat_idx = unique_categories.index(cat)
            sentiment_matrix[cat_idx, sent] += 1
        
        # 归一化
        row_sums = sentiment_matrix.sum(axis=1, keepdims=True)
        sentiment_matrix_norm = np.divide(sentiment_matrix, row_sums, 
                                         where=row_sums != 0)
        
        # 创建热力图
        plt.figure(figsize=(8, max(6, len(unique_categories) * 0.3)))
        sns.heatmap(sentiment_matrix_norm, 
                   annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=['负面', '正面'],
                   yticklabels=unique_categories,
                   cbar_kws={'label': '比例'})
        plt.title('各类别情绪分布热力图')
        plt.xlabel('情绪')
        plt.ylabel('类别')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"情绪热力图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_sentiment_report(self, texts: List[str], sentiments: List[int],
                               probabilities: Optional[np.ndarray] = None,
                               save_dir: str = './visualization_report'):
        """
        创建完整的情绪分析可视化报告
        
        Args:
            texts: 文本列表
            sentiments: 情绪标签列表
            probabilities: 预测概率（可选）
            save_dir: 保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n生成可视化报告到: {save_dir}")
        print("-" * 50)
        
        # 1. 情绪分布
        self.plot_sentiment_distribution(
            sentiments, 
            save_path=os.path.join(save_dir, 'sentiment_distribution.png')
        )
        
        # 2. 情绪时间线
        self.plot_sentiment_timeline(
            list(range(len(sentiments))), sentiments,
            save_path=os.path.join(save_dir, 'sentiment_timeline.png')
        )
        
        # 3. 置信度分布（如果有概率）
        if probabilities is not None:
            self.plot_confidence_distribution(
                probabilities,
                true_labels=sentiments,
                save_path=os.path.join(save_dir, 'confidence_distribution.png')
            )
        
        # 4. 词云图
        try:
            # 正面词云
            self.create_wordcloud(
                texts, sentiments, sentiment_filter=1,
                save_path=os.path.join(save_dir, 'wordcloud_positive.png')
            )
            
            # 负面词云
            self.create_wordcloud(
                texts, sentiments, sentiment_filter=0,
                save_path=os.path.join(save_dir, 'wordcloud_negative.png')
            )
        except Exception as e:
            print(f"词云生成失败: {e}")
        
        print("\n可视化报告生成完成！")
    
    def _check_font(self, font_name: str) -> bool:
        """检查字体是否存在"""
        try:
            fm.findfont(font_name)
            return True
        except:
            return False