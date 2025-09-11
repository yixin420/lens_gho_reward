"""
情绪分析主程序
提供统一的接口进行情绪分析
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime

from preprocessor import TextPreprocessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from visualizer import SentimentVisualizer

# 尝试导入NLTK情感分析器
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK未安装，基于规则的分析将使用简化版本")


class SentimentAnalyzer:
    """统一的情绪分析器"""
    
    def __init__(self, method='ml', model_type='nb', language='zh', model_path=None):
        """
        初始化情绪分析器
        
        Args:
            method: 分析方法 'rule'(基于规则), 'ml'(机器学习), 'dl'(深度学习)
            model_type: 模型类型（用于ml方法）
            language: 语言类型
            model_path: 预训练模型路径
        """
        self.method = method
        self.model_type = model_type
        self.language = language
        
        # 初始化组件
        self.preprocessor = TextPreprocessor(language=language)
        self.evaluator = ModelEvaluator()
        self.visualizer = SentimentVisualizer(language=language)
        
        # 初始化模型
        if method == 'rule':
            self._init_rule_based()
        elif method == 'ml':
            self._init_ml_model(model_type, model_path)
        elif method == 'dl':
            self._init_dl_model(model_path)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 分析历史
        self.history = []
    
    def _init_rule_based(self):
        """初始化基于规则的分析器"""
        if NLTK_AVAILABLE and self.language == 'en':
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.rule_analyzer = SentimentIntensityAnalyzer()
            except:
                self.rule_analyzer = None
        else:
            self.rule_analyzer = None
        
        # 情感词典
        self._init_sentiment_lexicon()
    
    def _init_sentiment_lexicon(self):
        """初始化情感词典"""
        if self.language == 'zh':
            self.positive_words = set([
                '好', '棒', '优秀', '喜欢', '爱', '开心', '快乐', '幸福', '美好', '精彩',
                '完美', '赞', '不错', '很好', '太好了', '真好', '最好', '超级', '非常好',
                '满意', '高兴', '愉快', '欣喜', '兴奋', '激动', '感动', '温暖', '舒服',
                '优雅', '漂亮', '美丽', '可爱', '迷人', '出色', '卓越', '杰出', '优异'
            ])
            
            self.negative_words = set([
                '差', '糟糕', '坏', '讨厌', '恨', '难过', '伤心', '失望', '糟', '烂',
                '垃圾', '无聊', '恶心', '可怕', '痛苦', '悲伤', '愤怒', '生气', '不好',
                '最差', '太差', '很差', '不满', '失落', '沮丧', '郁闷', '烦躁', '焦虑',
                '难看', '丑陋', '恶劣', '低劣', '粗糙', '肮脏', '令人失望', '让人失望'
            ])
            
            # 否定词
            self.negation_words = set(['不', '没', '无', '非', '别', '未', '否', '难'])
            
            # 程度副词
            self.intensifiers = {
                '很': 1.5, '非常': 2.0, '特别': 2.0, '极其': 2.5, '最': 3.0,
                '太': 2.0, '超': 2.0, '格外': 1.8, '相当': 1.5, '十分': 1.8
            }
        else:
            self.positive_words = set([
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
                'like', 'happy', 'joy', 'beautiful', 'perfect', 'best', 'awesome', 'nice',
                'positive', 'fortunate', 'correct', 'superior', 'fun', 'exciting', 'brilliant',
                'outstanding', 'magnificent', 'marvelous', 'superb', 'terrific', 'fabulous'
            ])
            
            self.negative_words = set([
                'bad', 'terrible', 'horrible', 'awful', 'hate', 'dislike', 'sad', 'angry',
                'disappointed', 'poor', 'worst', 'boring', 'disgusting', 'ugly', 'nasty',
                'negative', 'unfortunate', 'wrong', 'inferior', 'annoying', 'frustrating',
                'dreadful', 'appalling', 'atrocious', 'abysmal', 'pathetic', 'miserable'
            ])
            
            self.negation_words = set(['not', 'no', 'never', 'neither', 'nor', 'none'])
            
            self.intensifiers = {
                'very': 1.5, 'extremely': 2.0, 'absolutely': 2.5, 'completely': 2.0,
                'totally': 2.0, 'really': 1.5, 'quite': 1.3, 'pretty': 1.3
            }
    
    def _init_ml_model(self, model_type, model_path):
        """初始化机器学习模型"""
        self.trainer = ModelTrainer(model_type=model_type, language=self.language)
        
        if model_path and os.path.exists(model_path):
            self.trainer.load_model(model_path)
            print(f"已加载模型: {model_path}")
        else:
            print(f"使用新的{model_type.upper()}模型，需要训练后才能使用")
    
    def _init_dl_model(self, model_path):
        """初始化深度学习模型"""
        try:
            from model_trainer import DeepLearningTrainer
            self.dl_trainer = DeepLearningTrainer()
            
            if model_path and os.path.exists(model_path):
                self.dl_trainer.load_model(model_path)
                print(f"已加载深度学习模型: {model_path}")
        except ImportError:
            raise ImportError("深度学习库未安装，请安装transformers和torch")
    
    def analyze(self, text: str) -> Dict:
        """
        分析单个文本的情绪
        
        Args:
            text: 输入文本
            
        Returns:
            分析结果字典
        """
        result = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'method': self.method
        }
        
        if self.method == 'rule':
            result.update(self._analyze_rule_based(text))
        elif self.method == 'ml':
            result.update(self._analyze_ml(text))
        elif self.method == 'dl':
            result.update(self._analyze_dl(text))
        
        # 保存到历史
        self.history.append(result)
        
        return result
    
    def _analyze_rule_based(self, text: str) -> Dict:
        """基于规则的情绪分析"""
        # 预处理文本
        processed_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize(text)
        
        if self.rule_analyzer and self.language == 'en':
            # 使用NLTK VADER
            scores = self.rule_analyzer.polarity_scores(text)
            sentiment = 1 if scores['compound'] >= 0.05 else 0
            confidence = abs(scores['compound'])
            
            return {
                'sentiment': sentiment,
                'sentiment_label': '正面' if sentiment == 1 else '负面',
                'confidence': confidence,
                'scores': scores,
                'method_details': 'NLTK VADER'
            }
        else:
            # 使用自定义规则
            positive_score = 0
            negative_score = 0
            
            # 检查否定词
            has_negation = any(word in tokens for word in self.negation_words)
            
            for i, token in enumerate(tokens):
                # 检查程度副词
                intensifier = 1.0
                if i > 0 and tokens[i-1] in self.intensifiers:
                    intensifier = self.intensifiers[tokens[i-1]]
                
                # 计算情感分数
                if token in self.positive_words:
                    if has_negation:
                        negative_score += intensifier
                    else:
                        positive_score += intensifier
                elif token in self.negative_words:
                    if has_negation:
                        positive_score += intensifier
                    else:
                        negative_score += intensifier
            
            # 计算最终情感
            total_score = positive_score - negative_score
            sentiment = 1 if total_score > 0 else 0
            confidence = abs(total_score) / max(len(tokens), 1)
            confidence = min(confidence, 1.0)  # 限制在0-1范围
            
            return {
                'sentiment': sentiment,
                'sentiment_label': '正面' if sentiment == 1 else '负面',
                'confidence': confidence,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'method_details': 'Custom Rule-based'
            }
    
    def _analyze_ml(self, text: str) -> Dict:
        """机器学习模型分析"""
        if not hasattr(self.trainer, 'model') or self.trainer.model is None:
            raise ValueError("机器学习模型未训练，请先训练模型")
        
        # 预测
        sentiment = self.trainer.predict(text)
        probabilities = self.trainer.predict_proba(text)
        
        return {
            'sentiment': int(sentiment),
            'sentiment_label': '正面' if sentiment == 1 else '负面',
            'confidence': float(max(probabilities)),
            'probabilities': {
                '负面': float(probabilities[0]),
                '正面': float(probabilities[1])
            },
            'method_details': f'ML-{self.model_type.upper()}'
        }
    
    def _analyze_dl(self, text: str) -> Dict:
        """深度学习模型分析"""
        if not hasattr(self, 'dl_trainer'):
            raise ValueError("深度学习模型未初始化")
        
        # 预测
        sentiment = self.dl_trainer.predict(text)
        
        return {
            'sentiment': sentiment,
            'sentiment_label': '正面' if sentiment == 1 else '负面',
            'confidence': 0.95,  # 深度学习模型通常有较高置信度
            'method_details': 'Deep Learning (BERT)'
        }
    
    def batch_analyze(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        批量分析文本情绪
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            分析结果列表
        """
        results = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            result = self.analyze(text)
            results.append(result)
        
        if show_progress:
            print(f"批量分析完成: {total}个文本")
        
        return results
    
    def train(self, train_data: Dict, val_data: Optional[Dict] = None, **kwargs):
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            **kwargs: 其他训练参数
        """
        if self.method == 'rule':
            print("基于规则的方法不需要训练")
            return
        
        if self.method == 'ml':
            metrics = self.trainer.train(train_data, val_data, **kwargs)
            return metrics
        elif self.method == 'dl':
            metrics = self.dl_trainer.train(train_data, val_data, **kwargs)
            return metrics
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        """
        评估模型性能
        
        Args:
            texts: 测试文本
            labels: 真实标签
            
        Returns:
            评估结果
        """
        # 批量预测
        predictions = []
        probabilities = []
        
        for text in texts:
            result = self.analyze(text)
            predictions.append(result['sentiment'])
            
            if 'probabilities' in result:
                probabilities.append([result['probabilities']['负面'], 
                                     result['probabilities']['正面']])
        
        # 评估
        if probabilities:
            metrics = self.evaluator.evaluate(labels, predictions, 
                                             np.array(probabilities))
        else:
            metrics = self.evaluator.evaluate(labels, predictions)
        
        # 打印报告
        self.evaluator.print_report(labels, predictions)
        
        return metrics
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.method == 'rule':
            print("基于规则的方法不需要保存模型")
            return
        
        if self.method == 'ml':
            self.trainer.save_model(filepath)
        elif self.method == 'dl':
            self.dl_trainer.save_model(filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        if self.method == 'ml':
            self.trainer.load_model(filepath)
        elif self.method == 'dl':
            self.dl_trainer.load_model(filepath)
    
    def export_history(self, filepath: str):
        """导出分析历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"分析历史已导出到: {filepath}")
    
    def visualize_results(self, texts: Optional[List[str]] = None,
                         sentiments: Optional[List[int]] = None,
                         save_dir: str = './visualizations'):
        """
        可视化分析结果
        
        Args:
            texts: 文本列表（可选，默认使用历史记录）
            sentiments: 情绪标签列表（可选，默认使用历史记录）
            save_dir: 保存目录
        """
        if texts is None and self.history:
            texts = [h['text'] for h in self.history]
            sentiments = [h['sentiment'] for h in self.history]
        
        if texts and sentiments:
            self.visualizer.create_sentiment_report(texts, sentiments, save_dir=save_dir)
        else:
            print("没有可视化的数据")


class SentimentPipeline:
    """情绪分析流水线"""
    
    def __init__(self, config: Dict):
        """
        初始化流水线
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.analyzers = {}
        
        # 初始化多个分析器
        for name, settings in config.get('analyzers', {}).items():
            self.analyzers[name] = SentimentAnalyzer(**settings)
    
    def run(self, texts: List[str]) -> Dict:
        """
        运行完整的分析流水线
        
        Args:
            texts: 文本列表
            
        Returns:
            综合分析结果
        """
        results = {}
        
        for name, analyzer in self.analyzers.items():
            print(f"\n使用 {name} 分析器...")
            results[name] = analyzer.batch_analyze(texts, show_progress=False)
        
        # 综合结果
        combined_results = self._combine_results(results)
        
        return combined_results
    
    def _combine_results(self, results: Dict) -> Dict:
        """组合多个分析器的结果"""
        combined = {
            'individual_results': results,
            'ensemble_prediction': [],
            'confidence_scores': []
        }
        
        # 投票集成
        num_texts = len(list(results.values())[0])
        
        for i in range(num_texts):
            votes = []
            confidences = []
            
            for analyzer_results in results.values():
                votes.append(analyzer_results[i]['sentiment'])
                confidences.append(analyzer_results[i]['confidence'])
            
            # 多数投票
            ensemble_sentiment = 1 if sum(votes) > len(votes) / 2 else 0
            avg_confidence = np.mean(confidences)
            
            combined['ensemble_prediction'].append(ensemble_sentiment)
            combined['confidence_scores'].append(avg_confidence)
        
        return combined