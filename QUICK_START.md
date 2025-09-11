# 情绪分析系统 - 快速开始指南

## 系统概述

这是一个功能完整的中英文情绪分析系统，提供多种分析方法和可视化功能。

### 主要特性

✅ **多语言支持**
- 中文情绪分析
- 英文情绪分析

✅ **多种分析方法**
- 基于规则的情绪分析
- 机器学习模型（朴素贝叶斯、SVM、逻辑回归等）
- 深度学习模型（BERT，需要额外安装）

✅ **完整的功能模块**
- 数据预处理
- 模型训练
- 模型评估
- 结果可视化
- 批量处理

## 快速安装

```bash
# 安装基础依赖
pip3 install --break-system-packages -r requirements.txt

# 下载NLTK数据
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## 快速使用

### 1. 基础情绪分析

```python
from sentiment_analyzer import SentimentAnalyzer

# 创建分析器
analyzer = SentimentAnalyzer(method='rule', language='zh')

# 分析文本
text = "这个产品真的太棒了！"
result = analyzer.analyze(text)

print(f"情绪: {result['sentiment_label']}")  # 输出: 情绪: 正面
print(f"置信度: {result['confidence']}")      # 输出: 置信度: 0.XX
```

### 2. 批量分析

```python
texts = [
    "很开心",
    "太糟糕了",
    "还可以"
]

results = analyzer.batch_analyze(texts)
for r in results:
    print(f"{r['text']}: {r['sentiment_label']}")
```

### 3. 训练自定义模型

```python
# 准备训练数据
train_data = {
    'texts': ["好产品", "差评", ...],
    'labels': [1, 0, ...]  # 1=正面, 0=负面
}

# 训练模型
analyzer = SentimentAnalyzer(method='ml', model_type='svm', language='zh')
analyzer.train(train_data)

# 保存模型
analyzer.save_model('my_model.pkl')
```

## 运行示例

系统提供了多个示例脚本：

```bash
# 基础示例
python3 examples/basic_example.py

# 模型训练示例
python3 examples/train_model_example.py

# 批量分析示例
python3 examples/batch_analysis.py

# 可视化示例
python3 examples/visualization_example.py
```

## 项目结构

```
/workspace/
├── sentiment_analyzer.py    # 主分析器
├── preprocessor.py         # 数据预处理
├── model_trainer.py        # 模型训练
├── evaluator.py           # 模型评估
├── visualizer.py          # 结果可视化
├── test_sentiment.py      # 测试脚本
├── examples/              # 示例代码
│   ├── basic_example.py
│   ├── train_model_example.py
│   ├── batch_analysis.py
│   └── visualization_example.py
├── data/                  # 示例数据
│   └── sample_reviews.json
├── models/                # 保存的模型
├── requirements.txt       # 依赖列表
└── README.md             # 详细文档
```

## API 参考

### SentimentAnalyzer 类

主要方法：
- `analyze(text)` - 分析单个文本
- `batch_analyze(texts)` - 批量分析
- `train(data)` - 训练模型
- `evaluate(texts, labels)` - 评估模型
- `save_model(path)` - 保存模型
- `load_model(path)` - 加载模型
- `visualize_results()` - 可视化结果

### 支持的模型类型

- `'nb'` - 朴素贝叶斯
- `'svm'` - 支持向量机
- `'lr'` - 逻辑回归
- `'rf'` - 随机森林
- `'gb'` - 梯度提升

## 性能指标

系统提供详细的性能评估：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数
- 混淆矩阵
- ROC曲线
- PR曲线

## 注意事项

1. 中文分析使用jieba分词，首次运行会构建词典缓存
2. 英文分析使用NLTK，需要下载相关数据包
3. 深度学习功能需要额外安装transformers和torch
4. 可视化功能需要中文字体支持（如遇问题可修改字体设置）

## 常见问题

**Q: 中文分词不准确怎么办？**
A: 可以通过jieba添加自定义词典来提高分词准确性。

**Q: 如何提高分析准确率？**
A: 使用更多的训练数据，或尝试不同的模型类型。

**Q: 支持其他语言吗？**
A: 目前主要支持中英文，可以通过修改预处理器来支持其他语言。

## 联系和支持

如有问题或建议，请查看项目README.md文件获取更多信息。

---

快速开始，享受情绪分析的乐趣！🎉