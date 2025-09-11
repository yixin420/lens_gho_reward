"""
数据预处理模块
支持中英文文本的预处理，包括分词、去停用词、特征提取等
"""

import re
import jieba
import numpy as np
from typing import List, Dict, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, language='zh', remove_stopwords=True, lowercase=True):
        """
        初始化预处理器
        
        Args:
            language: 语言类型 'zh'中文 或 'en'英文
            remove_stopwords: 是否去除停用词
            lowercase: 是否转换为小写
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        # 初始化停用词
        self._init_stopwords()
        
        # 初始化向量化器
        self.vectorizer = None
        
    def _init_stopwords(self):
        """初始化停用词表"""
        if self.language == 'zh':
            # 中文停用词
            self.stopwords = set([
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                '自己', '这', '那', '些', '吗', '呢', '吧', '啊', '哦', '呀', '嗯', '哈'
            ])
        else:
            # 英文停用词
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                # 如果NLTK数据未下载，使用基础停用词
                self.stopwords = set([
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                    'into', 'through', 'during', 'before', 'after', 'above', 'below',
                    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under'
                ])
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        if self.language == 'zh':
            # 移除非中文字符（保留基本标点）
            text = re.sub(r'[^\u4e00-\u9fa5\s，。！？；：、""''（）《》【】…—]', '', text)
        else:
            # 移除特殊字符（保留字母、数字和基本标点）
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
            
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 转换为小写
        if self.lowercase:
            text = text.lower()
            
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 文本
            
        Returns:
            分词结果
        """
        if self.language == 'zh':
            # 中文分词
            tokens = list(jieba.cut(text))
        else:
            # 英文分词
            tokens = word_tokenize(text)
            
        # 去除停用词
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords and token.strip()]
            
        return tokens
    
    def preprocess(self, text: str) -> str:
        """
        完整的预处理流程
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 清洗文本
        text = self.clean_text(text)
        
        # 分词
        tokens = self.tokenize(text)
        
        # 重新组合为字符串
        return ' '.join(tokens)
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        批量预处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            预处理后的文本列表
        """
        return [self.preprocess(text) for text in texts]
    
    def extract_features(self, texts: List[str], method='tfidf', max_features=5000) -> np.ndarray:
        """
        提取文本特征
        
        Args:
            texts: 文本列表
            method: 特征提取方法 'tfidf' 或 'count'
            max_features: 最大特征数
            
        Returns:
            特征矩阵
        """
        # 预处理文本
        processed_texts = self.batch_preprocess(texts)
        
        # 选择向量化器
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)
            
        # 提取特征
        features = self.vectorizer.fit_transform(processed_texts)
        
        return features.toarray()
    
    def transform_features(self, texts: List[str]) -> np.ndarray:
        """
        使用已训练的向量化器转换文本
        
        Args:
            texts: 文本列表
            
        Returns:
            特征矩阵
        """
        if self.vectorizer is None:
            raise ValueError("向量化器未初始化，请先调用extract_features方法")
            
        processed_texts = self.batch_preprocess(texts)
        features = self.vectorizer.transform(processed_texts)
        
        return features.toarray()
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        提取情感特征
        
        Args:
            text: 文本
            
        Returns:
            情感特征字典
        """
        features = {}
        
        # 感叹号数量
        features['exclamation_count'] = text.count('!')
        features['exclamation_ratio'] = text.count('!') / max(len(text), 1)
        
        # 问号数量
        features['question_count'] = text.count('?')
        features['question_ratio'] = text.count('?') / max(len(text), 1)
        
        # 大写字母比例（仅英文）
        if self.language == 'en':
            uppercase_count = sum(1 for c in text if c.isupper())
            features['uppercase_ratio'] = uppercase_count / max(len(text), 1)
        
        # 文本长度
        features['text_length'] = len(text)
        features['word_count'] = len(self.tokenize(text))
        
        # 情感词汇
        positive_words = self._get_positive_words()
        negative_words = self._get_negative_words()
        
        tokens = self.tokenize(text.lower())
        features['positive_word_count'] = sum(1 for token in tokens if token in positive_words)
        features['negative_word_count'] = sum(1 for token in tokens if token in negative_words)
        features['sentiment_ratio'] = (features['positive_word_count'] - features['negative_word_count']) / max(len(tokens), 1)
        
        return features
    
    def _get_positive_words(self) -> set:
        """获取积极词汇表"""
        if self.language == 'zh':
            return set([
                '好', '棒', '优秀', '喜欢', '爱', '开心', '快乐', '幸福', '美好', '精彩',
                '完美', '赞', '不错', '很好', '太好了', '真好', '最好', '超级', '非常好',
                '满意', '高兴', '愉快', '欣喜', '兴奋', '激动', '感动', '温暖', '舒服'
            ])
        else:
            return set([
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
                'like', 'happy', 'joy', 'beautiful', 'perfect', 'best', 'awesome', 'nice',
                'positive', 'fortunate', 'correct', 'superior', 'fun', 'exciting', 'brilliant'
            ])
    
    def _get_negative_words(self) -> set:
        """获取消极词汇表"""
        if self.language == 'zh':
            return set([
                '差', '糟糕', '坏', '讨厌', '恨', '难过', '伤心', '失望', '糟', '烂',
                '垃圾', '无聊', '恶心', '可怕', '痛苦', '悲伤', '愤怒', '生气', '不好',
                '最差', '太差', '很差', '不满', '失落', '沮丧', '郁闷', '烦躁', '焦虑'
            ])
        else:
            return set([
                'bad', 'terrible', 'horrible', 'awful', 'hate', 'dislike', 'sad', 'angry',
                'disappointed', 'poor', 'worst', 'boring', 'disgusting', 'ugly', 'nasty',
                'negative', 'unfortunate', 'wrong', 'inferior', 'annoying', 'frustrating'
            ])


class DataAugmenter:
    """数据增强器"""
    
    def __init__(self, language='zh'):
        """
        初始化数据增强器
        
        Args:
            language: 语言类型
        """
        self.language = language
        
    def synonym_replacement(self, text: str, n=2) -> str:
        """
        同义词替换
        
        Args:
            text: 原始文本
            n: 替换数量
            
        Returns:
            增强后的文本
        """
        # 这里简化处理，实际应用中可以使用词向量或同义词词典
        return text
    
    def random_insertion(self, text: str, n=1) -> str:
        """
        随机插入
        
        Args:
            text: 原始文本
            n: 插入数量
            
        Returns:
            增强后的文本
        """
        # 简化处理
        return text
    
    def random_swap(self, text: str, n=1) -> str:
        """
        随机交换
        
        Args:
            text: 原始文本
            n: 交换次数
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def random_deletion(self, text: str, p=0.1) -> str:
        """
        随机删除
        
        Args:
            text: 原始文本
            p: 删除概率
            
        Returns:
            增强后的文本
        """
        words = text.split()
        if len(words) == 1:
            return text
            
        new_words = [word for word in words if np.random.random() > p]
        
        # 确保至少保留一个词
        if len(new_words) == 0:
            return words[np.random.randint(len(words))]
            
        return ' '.join(new_words)
    
    def augment(self, text: str, num_aug=1) -> List[str]:
        """
        数据增强
        
        Args:
            text: 原始文本
            num_aug: 生成增强样本数量
            
        Returns:
            增强后的文本列表
        """
        augmented_texts = []
        
        for _ in range(num_aug):
            aug_text = text
            
            # 随机选择增强方法
            methods = [self.random_swap, self.random_deletion]
            method = np.random.choice(methods)
            
            aug_text = method(aug_text)
            augmented_texts.append(aug_text)
            
        return augmented_texts