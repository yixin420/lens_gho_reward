"""
基础示例：展示情绪分析的基本用法
"""

import sys
sys.path.append('..')

from sentiment_analyzer import SentimentAnalyzer

def main():
    print("="*60)
    print("情绪分析基础示例")
    print("="*60)
    
    # 测试文本
    test_texts_zh = [
        "这部电影真的太棒了！我非常喜欢！",
        "服务态度很差，完全不推荐。",
        "产品质量还可以，价格也合理。",
        "今天心情特别好，一切都很顺利！",
        "这个餐厅的食物难吃死了，再也不会来了。",
        "虽然有些小问题，但总体来说还是很满意的。"
    ]
    
    test_texts_en = [
        "This movie is absolutely fantastic! I love it!",
        "The service was terrible, would not recommend.",
        "The product quality is okay, price is reasonable.",
        "I'm feeling great today, everything is going well!",
        "The food at this restaurant is awful, never coming back.",
        "Despite some minor issues, overall I'm quite satisfied."
    ]
    
    # 1. 基于规则的分析（中文）
    print("\n1. 基于规则的情绪分析（中文）")
    print("-"*40)
    
    analyzer_rule_zh = SentimentAnalyzer(method='rule', language='zh')
    
    for text in test_texts_zh:
        result = analyzer_rule_zh.analyze(text)
        print(f"\n文本: {text}")
        print(f"情绪: {result['sentiment_label']}")
        print(f"置信度: {result['confidence']:.2f}")
        if 'positive_score' in result:
            print(f"正面分数: {result['positive_score']:.2f}")
            print(f"负面分数: {result['negative_score']:.2f}")
    
    # 2. 基于规则的分析（英文）
    print("\n\n2. 基于规则的情绪分析（英文）")
    print("-"*40)
    
    analyzer_rule_en = SentimentAnalyzer(method='rule', language='en')
    
    for text in test_texts_en:
        result = analyzer_rule_en.analyze(text)
        print(f"\n文本: {text}")
        print(f"情绪: {result['sentiment_label']}")
        print(f"置信度: {result['confidence']:.2f}")
    
    # 3. 批量分析
    print("\n\n3. 批量分析示例")
    print("-"*40)
    
    batch_results = analyzer_rule_zh.batch_analyze(test_texts_zh, show_progress=False)
    
    positive_count = sum(1 for r in batch_results if r['sentiment'] == 1)
    negative_count = len(batch_results) - positive_count
    
    print(f"分析了 {len(batch_results)} 个文本")
    print(f"正面情绪: {positive_count} 个")
    print(f"负面情绪: {negative_count} 个")
    print(f"平均置信度: {sum(r['confidence'] for r in batch_results) / len(batch_results):.2f}")
    
    # 4. 导出分析历史
    print("\n\n4. 导出分析历史")
    print("-"*40)
    
    analyzer_rule_zh.export_history('analysis_history.json')
    
    print("\n示例运行完成！")

if __name__ == "__main__":
    main()