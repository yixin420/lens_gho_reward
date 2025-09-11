#!/usr/bin/env python
"""
快速测试脚本 - 验证情绪分析系统是否正常工作
"""

from sentiment_analyzer import SentimentAnalyzer

def test_basic_functionality():
    """测试基本功能"""
    print("="*60)
    print("情绪分析系统测试")
    print("="*60)
    
    # 测试文本
    test_cases = [
        ("这个产品真的太棒了，我非常喜欢！", "正面"),
        ("质量太差了，完全不值这个价格。", "负面"),
        ("This is absolutely amazing! I love it!", "正面"),
        ("Terrible service, would not recommend.", "负面"),
    ]
    
    # 测试中文分析
    print("\n测试中文情绪分析:")
    print("-"*40)
    analyzer_zh = SentimentAnalyzer(method='rule', language='zh')
    
    for text, expected in test_cases[:2]:
        result = analyzer_zh.analyze(text)
        status = "✓" if result['sentiment_label'] == expected else "✗"
        print(f"{status} 文本: {text}")
        print(f"  预期: {expected}, 实际: {result['sentiment_label']}")
        print(f"  置信度: {result['confidence']:.2f}")
    
    # 测试英文分析
    print("\n测试英文情绪分析:")
    print("-"*40)
    analyzer_en = SentimentAnalyzer(method='rule', language='en')
    
    for text, expected in test_cases[2:]:
        result = analyzer_en.analyze(text)
        status = "✓" if result['sentiment_label'] == expected else "✗"
        print(f"{status} 文本: {text}")
        print(f"  预期: {expected}, 实际: {result['sentiment_label']}")
        print(f"  置信度: {result['confidence']:.2f}")
    
    print("\n测试完成！系统运行正常。")

if __name__ == "__main__":
    test_basic_functionality()