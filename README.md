# 情绪分析系统

一个支持中英文的情绪分析系统，提供多种模型和分析方法。

## 功能特点

- 支持中英文文本情绪分析
- 提供多种分析方法：
  - 基于规则的情绪分析
  - 基于机器学习的情绪分析（朴素贝叶斯、SVM）
  - 基于深度学习的情绪分析（BERT）
- 支持批量文本处理
- 提供可视化分析结果

## 安装依赖

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
```

## 使用方法

### 1. 基础使用

```python
from sentiment_analyzer import SentimentAnalyzer

# 创建分析器
analyzer = SentimentAnalyzer(method='ml', language='zh')

# 分析单个文本
text = "这部电影真的太棒了！"
result = analyzer.analyze(text)
print(result)

# 批量分析
texts = ["很开心", "太糟糕了", "还可以"]
results = analyzer.batch_analyze(texts)
```

### 2. 训练自定义模型

```python
from model_trainer import ModelTrainer

# 准备训练数据
train_data = {
    'texts': [...],
    'labels': [...]  # 1表示正面，0表示负面
}

# 训练模型
trainer = ModelTrainer(model_type='svm')
trainer.train(train_data)
trainer.save_model('my_model.pkl')
```

### 3. 运行示例

```bash
# 基础示例
python examples/basic_example.py

# 批量分析示例
python examples/batch_analysis.py

# 可视化示例
python examples/visualization_example.py
```

## 项目结构

```
sentiment-analysis/
├── sentiment_analyzer.py    # 主分析器
├── preprocessor.py          # 数据预处理
├── model_trainer.py         # 模型训练
├── evaluator.py            # 模型评估
├── visualizer.py           # 结果可视化
├── models/                 # 预训练模型
├── data/                   # 示例数据
└── examples/               # 使用示例
```