"""
模型训练模块
提供多种机器学习和深度学习模型的训练功能
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification, AdamW
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("深度学习库未安装，部分功能将不可用")

from preprocessor import TextPreprocessor


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_type='nb', language='zh', random_state=42):
        """
        初始化训练器
        
        Args:
            model_type: 模型类型 'nb'(朴素贝叶斯), 'svm', 'lr'(逻辑回归), 'rf'(随机森林), 'gb'(梯度提升)
            language: 语言类型
            random_state: 随机种子
        """
        self.model_type = model_type
        self.language = language
        self.random_state = random_state
        
        # 初始化预处理器
        self.preprocessor = TextPreprocessor(language=language)
        
        # 初始化模型
        self.model = self._init_model()
        
        # 训练历史
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        
    def _init_model(self):
        """初始化模型"""
        if self.model_type == 'nb':
            return MultinomialNB(alpha=1.0)
        elif self.model_type == 'svm':
            return SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state)
        elif self.model_type == 'lr':
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    test_size: float = 0.2, 
                    feature_method: str = 'tfidf') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            texts: 文本列表
            labels: 标签列表
            test_size: 测试集比例
            feature_method: 特征提取方法
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 提取特征
        X = self.preprocessor.extract_features(texts, method=feature_method)
        y = np.array(labels)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, train_data: Dict, val_data: Optional[Dict] = None, 
             feature_method: str = 'tfidf', cv_folds: int = 5) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            train_data: 训练数据字典 {'texts': [...], 'labels': [...]}
            val_data: 验证数据字典（可选）
            feature_method: 特征提取方法
            cv_folds: 交叉验证折数
            
        Returns:
            训练结果指标
        """
        texts = train_data['texts']
        labels = train_data['labels']
        
        if val_data is None:
            # 自动划分验证集
            X_train, X_val, y_train, y_val = self.prepare_data(texts, labels, feature_method=feature_method)
        else:
            # 使用提供的验证集
            X_train = self.preprocessor.extract_features(texts, method=feature_method)
            y_train = np.array(labels)
            X_val = self.preprocessor.transform_features(val_data['texts'])
            y_val = np.array(val_data['labels'])
        
        # 训练模型
        print(f"开始训练 {self.model_type.upper()} 模型...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # 计算指标
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'val_precision': precision_score(y_val, val_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'val_recall': recall_score(y_val, val_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'val_f1': f1_score(y_val, val_pred, average='weighted')
        }
        
        # 交叉验证
        if cv_folds > 1:
            X_all = self.preprocessor.extract_features(texts, method=feature_method)
            y_all = np.array(labels)
            cv_scores = cross_val_score(self.model, X_all, y_all, cv=cv_folds, scoring='accuracy')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            print(f"交叉验证准确率: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        # 保存历史
        self.history['train_acc'].append(metrics['train_accuracy'])
        self.history['val_acc'].append(metrics['val_accuracy'])
        
        # 打印结果
        print("\n训练结果:")
        print(f"训练集准确率: {metrics['train_accuracy']:.4f}")
        print(f"验证集准确率: {metrics['val_accuracy']:.4f}")
        print(f"训练集F1分数: {metrics['train_f1']:.4f}")
        print(f"验证集F1分数: {metrics['val_f1']:.4f}")
        
        return metrics
    
    def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        预测文本情感
        
        Args:
            texts: 文本或文本列表
            
        Returns:
            预测结果
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        # 提取特征
        X = self.preprocessor.transform_features(texts)
        
        # 预测
        predictions = self.model.predict(X)
        
        if single:
            return predictions[0]
        return predictions.tolist()
    
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        预测文本情感概率
        
        Args:
            texts: 文本或文本列表
            
        Returns:
            预测概率
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        
        # 提取特征
        X = self.preprocessor.transform_features(texts)
        
        # 预测概率
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        else:
            # 对于不支持概率预测的模型，返回硬预测
            predictions = self.model.predict(X)
            probabilities = np.zeros((len(predictions), 2))
            probabilities[np.arange(len(predictions)), predictions] = 1.0
        
        if single:
            return probabilities[0]
        return probabilities.tolist()
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # 保存模型和预处理器
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'language': self.language,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.model_type = model_data['model_type']
        self.language = model_data['language']
        self.history = model_data.get('history', {})
        
        print(f"模型已从 {filepath} 加载")


if DEEP_LEARNING_AVAILABLE:
    
    class TextDataset(Dataset):
        """PyTorch文本数据集"""
        
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    
    class DeepLearningTrainer:
        """深度学习模型训练器"""
        
        def __init__(self, model_name='bert-base-chinese', num_classes=2, device=None):
            """
            初始化深度学习训练器
            
            Args:
                model_name: 预训练模型名称
                num_classes: 分类数量
                device: 设备
            """
            self.model_name = model_name
            self.num_classes = num_classes
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 初始化tokenizer和模型
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_classes
            ).to(self.device)
            
            # 训练历史
            self.history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }
        
        def prepare_data(self, texts: List[str], labels: List[int], 
                        batch_size: int = 16, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
            """
            准备数据加载器
            
            Args:
                texts: 文本列表
                labels: 标签列表
                batch_size: 批次大小
                test_size: 测试集比例
                
            Returns:
                train_loader, val_loader
            """
            # 划分数据
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # 创建数据集
            train_dataset = TextDataset(X_train, y_train, self.tokenizer)
            val_dataset = TextDataset(X_val, y_val, self.tokenizer)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader
        
        def train(self, train_data: Dict, val_data: Optional[Dict] = None,
                 epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5) -> Dict[str, float]:
            """
            训练BERT模型
            
            Args:
                train_data: 训练数据
                val_data: 验证数据
                epochs: 训练轮数
                batch_size: 批次大小
                learning_rate: 学习率
                
            Returns:
                训练指标
            """
            texts = train_data['texts']
            labels = train_data['labels']
            
            # 准备数据
            train_loader, val_loader = self.prepare_data(texts, labels, batch_size)
            
            # 优化器
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
            
            # 训练循环
            print(f"开始训练BERT模型，设备: {self.device}")
            
            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.logits, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # 验证阶段
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        val_loss += outputs.loss.item()
                        
                        _, predicted = torch.max(outputs.logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # 计算指标
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                # 保存历史
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(avg_val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}")
                print(f"验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            return {
                'final_train_acc': self.history['train_acc'][-1],
                'final_val_acc': self.history['val_acc'][-1],
                'final_train_loss': self.history['train_loss'][-1],
                'final_val_loss': self.history['val_loss'][-1]
            }
        
        def predict(self, texts: Union[str, List[str]]) -> Union[int, List[int]]:
            """
            预测文本情感
            
            Args:
                texts: 文本或文本列表
                
            Returns:
                预测结果
            """
            if isinstance(texts, str):
                texts = [texts]
                single = True
            else:
                single = False
            
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for text in texts:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(**encoding)
                    _, predicted = torch.max(outputs.logits, 1)
                    predictions.append(predicted.item())
            
            if single:
                return predictions[0]
            return predictions
        
        def save_model(self, filepath: str):
            """保存模型"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': self.tokenizer,
                'history': self.history
            }, filepath)
            print(f"深度学习模型已保存到: {filepath}")
        
        def load_model(self, filepath: str):
            """加载模型"""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.tokenizer = checkpoint['tokenizer']
            self.history = checkpoint.get('history', {})
            print(f"深度学习模型已从 {filepath} 加载")