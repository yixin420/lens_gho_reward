"""
可视化示例：展示情绪分析结果的可视化
"""

import sys
sys.path.append('..')
import numpy as np

from sentiment_analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer

def generate_test_data():
    """生成测试数据用于可视化"""
    # 模拟评论数据
    texts = [
        "产品很好用，推荐",
        "质量太差了",
        "还可以，一般",
        "非常满意",
        "不推荐购买",
        "超出预期",
        "有点失望",
        "物超所值",
        "完全是垃圾",
        "很喜欢",
        "一般般吧",
        "强烈推荐",
        "买了后悔",
        "质量不错",
        "价格太贵",
        "很实用",
        "不值这个价",
        "孩子很喜欢",
        "有瑕疵",
        "完美"
    ] * 5  # 重复以增加数据量
    
    # 生成对应的情绪标签（简单规则）
    sentiments = []
    for text in texts:
        if any(word in text for word in ['好', '满意', '推荐', '喜欢', '完美', '不错', '实用', '超出']):
            sentiments.append(1)
        elif any(word in text for word in ['差', '垃圾', '失望', '后悔', '贵', '瑕疵']):
            sentiments.append(0)
        else:
            sentiments.append(np.random.choice([0, 1]))
    
    # 生成模拟的预测概率
    probabilities = []
    for sentiment in sentiments:
        if sentiment == 1:
            # 正面情绪，概率偏高
            prob_positive = np.random.uniform(0.6, 0.95)
        else:
            # 负面情绪，概率偏低
            prob_positive = np.random.uniform(0.05, 0.4)
        probabilities.append([1 - prob_positive, prob_positive])
    
    return texts, sentiments, np.array(probabilities)

def main():
    print("="*60)
    print("情绪分析可视化示例")
    print("="*60)
    
    # 1. 生成测试数据
    print("\n1. 准备数据")
    print("-"*40)
    
    texts, sentiments, probabilities = generate_test_data()
    print(f"样本数: {len(texts)}")
    print(f"正面情绪: {sum(sentiments)} 个")
    print(f"负面情绪: {len(sentiments) - sum(sentiments)} 个")
    
    # 2. 创建可视化器
    print("\n2. 创建可视化")
    print("-"*40)
    
    visualizer = SentimentVisualizer(language='zh')
    
    # 3. 情绪分布图
    print("\n生成情绪分布图...")
    visualizer.plot_sentiment_distribution(
        sentiments,
        save_path='visualizations/sentiment_distribution.png'
    )
    
    # 4. 情绪时间线
    print("生成情绪时间线图...")
    visualizer.plot_sentiment_timeline(
        list(range(len(sentiments))),
        sentiments,
        window_size=5,
        save_path='visualizations/sentiment_timeline.png'
    )
    
    # 5. 置信度分布
    print("生成置信度分布图...")
    visualizer.plot_confidence_distribution(
        probabilities,
        true_labels=sentiments,
        save_path='visualizations/confidence_distribution.png'
    )
    
    # 6. 词云图
    print("生成词云图...")
    try:
        # 正面词云
        visualizer.create_wordcloud(
            texts, sentiments, 
            sentiment_filter=1,
            save_path='visualizations/wordcloud_positive.png'
        )
        
        # 负面词云
        visualizer.create_wordcloud(
            texts, sentiments,
            sentiment_filter=0,
            save_path='visualizations/wordcloud_negative.png'
        )
    except Exception as e:
        print(f"词云生成失败（可能缺少中文字体）: {e}")
    
    # 7. 使用分析器生成完整报告
    print("\n3. 生成完整可视化报告")
    print("-"*40)
    
    analyzer = SentimentAnalyzer(method='rule', language='zh')
    
    # 分析一些文本
    sample_texts = texts[:20]
    for text in sample_texts:
        analyzer.analyze(text)
    
    # 生成可视化报告
    analyzer.visualize_results(save_dir='visualizations/full_report')
    
    print("\n可视化示例完成！")
    print("请查看 visualizations 目录中的图片文件")

if __name__ == "__main__":
    import os
    os.makedirs('visualizations', exist_ok=True)
    main()