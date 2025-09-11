"""
批量分析示例：展示如何处理大量文本数据
"""

import sys
sys.path.append('..')
import time
import json

from sentiment_analyzer import SentimentAnalyzer, SentimentPipeline

def load_sample_reviews():
    """加载示例评论数据"""
    # 电商评论示例
    reviews = [
        {"text": "手机很好用，拍照清晰，运行流畅", "category": "手机"},
        {"text": "电池续航太差了，用不了半天就没电", "category": "手机"},
        {"text": "屏幕显示效果很棒，色彩鲜艳", "category": "手机"},
        {"text": "系统经常卡顿，体验很差", "category": "手机"},
        {"text": "性价比很高，值得购买", "category": "手机"},
        
        {"text": "衣服质量很好，面料舒适", "category": "服装"},
        {"text": "尺码偏小，建议买大一号", "category": "服装"},
        {"text": "款式很时尚，很喜欢", "category": "服装"},
        {"text": "做工粗糙，线头很多", "category": "服装"},
        {"text": "颜色和图片一样，没有色差", "category": "服装"},
        
        {"text": "书的内容很精彩，值得一读", "category": "图书"},
        {"text": "印刷质量差，字迹模糊", "category": "图书"},
        {"text": "包装很好，没有损坏", "category": "图书"},
        {"text": "内容空洞，浪费时间", "category": "图书"},
        {"text": "作者文笔很好，引人入胜", "category": "图书"},
        
        {"text": "食物新鲜，味道很好", "category": "食品"},
        {"text": "已经过期了，不能吃", "category": "食品"},
        {"text": "包装精美，送人很合适", "category": "食品"},
        {"text": "口感很差，不推荐", "category": "食品"},
        {"text": "孩子很喜欢吃", "category": "食品"},
    ]
    
    # 扩展数据集
    extended_reviews = []
    for review in reviews * 3:  # 复制3次以增加数据量
        extended_reviews.append(review)
    
    return extended_reviews

def main():
    print("="*60)
    print("批量分析示例")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载评论数据")
    print("-"*40)
    
    reviews = load_sample_reviews()
    texts = [r['text'] for r in reviews]
    categories = [r['category'] for r in reviews]
    
    print(f"总评论数: {len(reviews)}")
    print(f"类别分布:")
    for cat in set(categories):
        count = categories.count(cat)
        print(f"  {cat}: {count} 条")
    
    # 2. 批量分析（基于规则）
    print("\n2. 基于规则的批量分析")
    print("-"*40)
    
    analyzer_rule = SentimentAnalyzer(method='rule', language='zh')
    
    start_time = time.time()
    results_rule = analyzer_rule.batch_analyze(texts, show_progress=True)
    end_time = time.time()
    
    print(f"\n分析完成！")
    print(f"处理时间: {end_time - start_time:.2f} 秒")
    print(f"平均每条: {(end_time - start_time) / len(texts) * 1000:.2f} 毫秒")
    
    # 3. 分类别统计
    print("\n3. 分类别情绪统计")
    print("-"*40)
    
    category_stats = {}
    for i, result in enumerate(results_rule):
        cat = categories[i]
        if cat not in category_stats:
            category_stats[cat] = {'positive': 0, 'negative': 0, 'total': 0}
        
        category_stats[cat]['total'] += 1
        if result['sentiment'] == 1:
            category_stats[cat]['positive'] += 1
        else:
            category_stats[cat]['negative'] += 1
    
    print(f"{'类别':<10} {'总数':<8} {'正面':<8} {'负面':<8} {'正面率':<10}")
    print("-"*44)
    for cat, stats in category_stats.items():
        positive_rate = stats['positive'] / stats['total'] * 100
        print(f"{cat:<10} {stats['total']:<8} {stats['positive']:<8} {stats['negative']:<8} {positive_rate:<10.1f}%")
    
    # 4. 找出最正面和最负面的评论
    print("\n4. 极端情绪评论")
    print("-"*40)
    
    # 按置信度排序
    sorted_results = sorted(enumerate(results_rule), 
                          key=lambda x: (x[1]['sentiment'], x[1]['confidence']), 
                          reverse=True)
    
    print("\n最正面的3条评论:")
    for idx, result in sorted_results[:3]:
        if result['sentiment'] == 1:
            print(f"  [{categories[idx]}] {texts[idx]}")
            print(f"    置信度: {result['confidence']:.2f}")
    
    print("\n最负面的3条评论:")
    for idx, result in sorted_results[-3:]:
        if result['sentiment'] == 0:
            print(f"  [{categories[idx]}] {texts[idx]}")
            print(f"    置信度: {result['confidence']:.2f}")
    
    # 5. 使用流水线进行多模型分析
    print("\n5. 多模型流水线分析")
    print("-"*40)
    
    # 配置流水线
    pipeline_config = {
        'analyzers': {
            '规则分析器': {
                'method': 'rule',
                'language': 'zh'
            }
        }
    }
    
    pipeline = SentimentPipeline(pipeline_config)
    
    # 运行流水线（使用部分数据）
    sample_texts = texts[:10]
    pipeline_results = pipeline.run(sample_texts)
    
    print("\n流水线分析结果:")
    for i, sentiment in enumerate(pipeline_results['ensemble_prediction']):
        label = '正面' if sentiment == 1 else '负面'
        confidence = pipeline_results['confidence_scores'][i]
        print(f"  文本 {i+1}: {label} (置信度: {confidence:.2f})")
    
    # 6. 导出结果
    print("\n6. 导出分析结果")
    print("-"*40)
    
    # 准备导出数据
    export_data = []
    for i, result in enumerate(results_rule):
        export_data.append({
            'text': texts[i],
            'category': categories[i],
            'sentiment': result['sentiment'],
            'sentiment_label': result['sentiment_label'],
            'confidence': result['confidence']
        })
    
    # 保存为JSON
    with open('batch_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print("结果已保存到 batch_analysis_results.json")
    
    # 7. 性能统计
    print("\n7. 性能统计")
    print("-"*40)
    
    total_positive = sum(1 for r in results_rule if r['sentiment'] == 1)
    total_negative = len(results_rule) - total_positive
    avg_confidence = sum(r['confidence'] for r in results_rule) / len(results_rule)
    
    print(f"总体统计:")
    print(f"  正面评论: {total_positive} ({total_positive/len(results_rule)*100:.1f}%)")
    print(f"  负面评论: {total_negative} ({total_negative/len(results_rule)*100:.1f}%)")
    print(f"  平均置信度: {avg_confidence:.3f}")
    
    print("\n批量分析示例完成！")

if __name__ == "__main__":
    main()