"""
模型训练示例：展示如何训练和使用机器学习模型
"""

import sys
sys.path.append('..')
import numpy as np

from sentiment_analyzer import SentimentAnalyzer

def generate_sample_data():
    """生成示例训练数据"""
    # 正面样本
    positive_texts = [
        "这个产品质量非常好，我很满意",
        "客服态度很好，解决问题很及时",
        "物流速度快，包装完好",
        "性价比很高，推荐购买",
        "使用体验非常棒",
        "超出预期，非常惊喜",
        "做工精细，材质优良",
        "功能强大，操作简单",
        "设计美观，很喜欢",
        "售后服务很好",
        "质量超乎想象的好",
        "完美符合我的需求",
        "非常实用，值得拥有",
        "颜色很正，没有色差",
        "孩子很喜欢，质量也好"
    ]
    
    # 负面样本
    negative_texts = [
        "质量太差了，很失望",
        "客服态度恶劣，不解决问题",
        "物流太慢，包装破损",
        "价格太贵，不值这个价",
        "使用体验很糟糕",
        "和描述不符，感觉被骗了",
        "做工粗糙，材质很差",
        "功能缺失，操作复杂",
        "设计丑陋，不实用",
        "没有售后服务",
        "质量问题严重",
        "完全不符合预期",
        "买了就后悔",
        "颜色差异很大",
        "产品有异味，不敢用"
    ]
    
    # 扩充数据（通过简单的变换）
    all_texts = []
    all_labels = []
    
    # 正面样本及其变体
    for text in positive_texts:
        all_texts.append(text)
        all_labels.append(1)
        # 添加变体
        all_texts.append(f"真的{text}")
        all_labels.append(1)
        all_texts.append(f"{text}，很推荐")
        all_labels.append(1)
    
    # 负面样本及其变体
    for text in negative_texts:
        all_texts.append(text)
        all_labels.append(0)
        # 添加变体
        all_texts.append(f"真的{text}")
        all_labels.append(0)
        all_texts.append(f"{text}，不推荐")
        all_labels.append(0)
    
    return {
        'texts': all_texts,
        'labels': all_labels
    }

def main():
    print("="*60)
    print("机器学习模型训练示例")
    print("="*60)
    
    # 1. 生成训练数据
    print("\n1. 准备训练数据")
    print("-"*40)
    
    train_data = generate_sample_data()
    print(f"训练样本数: {len(train_data['texts'])}")
    print(f"正面样本: {sum(train_data['labels'])} 个")
    print(f"负面样本: {len(train_data['labels']) - sum(train_data['labels'])} 个")
    
    # 2. 训练朴素贝叶斯模型
    print("\n2. 训练朴素贝叶斯模型")
    print("-"*40)
    
    analyzer_nb = SentimentAnalyzer(method='ml', model_type='nb', language='zh')
    metrics_nb = analyzer_nb.train(train_data, cv_folds=3)
    
    # 3. 训练SVM模型
    print("\n3. 训练SVM模型")
    print("-"*40)
    
    analyzer_svm = SentimentAnalyzer(method='ml', model_type='svm', language='zh')
    metrics_svm = analyzer_svm.train(train_data, cv_folds=3)
    
    # 4. 训练逻辑回归模型
    print("\n4. 训练逻辑回归模型")
    print("-"*40)
    
    analyzer_lr = SentimentAnalyzer(method='ml', model_type='lr', language='zh')
    metrics_lr = analyzer_lr.train(train_data, cv_folds=3)
    
    # 5. 模型比较
    print("\n5. 模型性能比较")
    print("-"*40)
    
    models_performance = {
        '朴素贝叶斯': {
            'accuracy': metrics_nb['val_accuracy'],
            'f1_score': metrics_nb['val_f1'],
            'cv_mean': metrics_nb.get('cv_mean', 0)
        },
        'SVM': {
            'accuracy': metrics_svm['val_accuracy'],
            'f1_score': metrics_svm['val_f1'],
            'cv_mean': metrics_svm.get('cv_mean', 0)
        },
        '逻辑回归': {
            'accuracy': metrics_lr['val_accuracy'],
            'f1_score': metrics_lr['val_f1'],
            'cv_mean': metrics_lr.get('cv_mean', 0)
        }
    }
    
    print("\n模型性能对比:")
    print(f"{'模型':<12} {'准确率':<10} {'F1分数':<10} {'交叉验证':<10}")
    print("-"*42)
    for model_name, perf in models_performance.items():
        print(f"{model_name:<12} {perf['accuracy']:<10.4f} {perf['f1_score']:<10.4f} {perf['cv_mean']:<10.4f}")
    
    # 6. 使用最佳模型进行预测
    print("\n6. 使用模型进行预测")
    print("-"*40)
    
    test_texts = [
        "这个东西质量真好，超级满意！",
        "太差了，完全是垃圾",
        "还行吧，一般般",
        "物超所值，强烈推荐",
        "有点小贵，但质量不错"
    ]
    
    print("\n使用朴素贝叶斯模型预测:")
    for text in test_texts:
        result = analyzer_nb.analyze(text)
        print(f"文本: {text}")
        print(f"  -> 情绪: {result['sentiment_label']}, 置信度: {result['confidence']:.2f}")
    
    # 7. 保存模型
    print("\n7. 保存模型")
    print("-"*40)
    
    analyzer_nb.save_model('models/nb_model.pkl')
    analyzer_svm.save_model('models/svm_model.pkl')
    analyzer_lr.save_model('models/lr_model.pkl')
    
    print("\n模型训练和保存完成！")

if __name__ == "__main__":
    main()