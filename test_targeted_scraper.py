#!/usr/bin/env python3
"""
测试针对性爬虫解决方案
基于之前的测试反馈进行验证
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_targeted_approach():
    """测试针对性方法"""
    logger.info("=" * 60)
    logger.info("测试针对性爬虫方法")
    logger.info("=" * 60)
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("访问长毛象网站...")
        driver.get("https://m.cmx.im/public/local")
        
        # 等待页面加载
        time.sleep(10)
        
        # 检查页面标题
        title = driver.title
        logger.info(f"页面标题: {title}")
        
        # 多次滚动加载内容
        logger.info("开始滚动加载内容...")
        for i in range(5):
            logger.info(f"第 {i+1} 次滚动...")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            # 检查当前有多少文本元素
            all_divs = driver.find_elements(By.TAG_NAME, "div")
            text_divs = [div for div in all_divs if div.text.strip() and len(div.text.strip()) > 20]
            logger.info(f"   找到 {len(text_divs)} 个有文本的div元素")
        
        # 分析找到的文本内容
        logger.info("分析文本内容...")
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_elements = []
        
        for div in text_divs[:20]:  # 只分析前20个
            text = div.text.strip()
            chinese_chars = chinese_pattern.findall(text)
            chinese_char_count = sum(len(chars) for chars in chinese_chars)
            
            if chinese_char_count > 0:
                chinese_elements.append({
                    'text': text[:200] + "..." if len(text) > 200 else text,
                    'chinese_chars': chinese_char_count,
                    'total_chars': len(text),
                    'ratio': chinese_char_count / len(text) if len(text) > 0 else 0
                })
        
        logger.info(f"找到 {len(chinese_elements)} 个包含中文的元素:")
        
        for i, element in enumerate(chinese_elements[:5], 1):  # 显示前5个
            logger.info(f"  {i}. 中文字符: {element['chinese_chars']}, 比例: {element['ratio']:.2%}")
            logger.info(f"     内容预览: {element['text'][:100]}...")
            logger.info("")
        
        # 测试不同的中文检测阈值
        logger.info("测试不同的中文检测阈值:")
        
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3]
        for threshold in thresholds:
            count = sum(1 for el in chinese_elements if el['ratio'] > threshold or el['chinese_chars'] > 1)
            logger.info(f"  阈值 {threshold:.0%}: {count} 个元素通过")
        
        # 保存调试信息
        with open("test_targeted_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        
        driver.save_screenshot("test_targeted_screenshot.png")
        
        logger.info("✓ 调试文件已保存")
        logger.info(f"✓ 测试完成，找到 {len(chinese_elements)} 个中文内容元素")
        
        driver.quit()
        
        return len(chinese_elements) > 0
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


def test_content_extraction():
    """测试内容提取逻辑"""
    logger.info("=" * 60)
    logger.info("测试内容提取逻辑")
    logger.info("=" * 60)
    
    # 模拟从测试中获得的文本内容
    test_texts = [
        "m.cmx.im 是可用于参与联邦宇宙的众多独立 Mastodon 站点之一。\n长毛象中文站是一个开放、友好、有爱的社区。长毛象中文站主题为喵、汪、各种动物、社交、科技、编程及生活。我们欢迎友好、热情...",
        "@用户名\n今天天气真好，适合出去走走！\n2小时前",
        "转发了一条消息\n这是一个很有趣的技术分享\n1天前",
        "Hello world! 这是一个测试帖子 😊\n刚刚",
        "纯英文内容 This is pure English content without Chinese",
        "短内容",
        "",
        "很长的中文内容，包含了很多信息。这里有技术讨论，生活分享，还有一些有趣的观点。希望大家都能在这个社区里找到自己感兴趣的内容。"
    ]
    
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    def is_chinese_content(text, ratio_threshold=0.05, char_threshold=1):
        if not text or len(text.strip()) < 2:
            return False
        
        chinese_chars = chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        ratio = chinese_char_count / total_chars
        return chinese_char_count >= char_threshold or ratio > ratio_threshold
    
    logger.info("测试中文内容检测:")
    passed = 0
    
    for i, text in enumerate(test_texts, 1):
        is_chinese = is_chinese_content(text)
        preview = text.replace('\n', ' ')[:80] + "..." if len(text) > 80 else text.replace('\n', ' ')
        
        logger.info(f"  {i}. 中文: {is_chinese} - {preview}")
        
        # 简单的期望判断
        expected = len(chinese_pattern.findall(text)) > 0
        if is_chinese == expected:
            passed += 1
    
    logger.info(f"中文检测测试: {passed}/{len(test_texts)} 通过")
    
    return passed >= len(test_texts) * 0.8  # 80%通过率


def main():
    """主测试函数"""
    print("针对性爬虫解决方案测试")
    print("=" * 50)
    
    tests = [
        ("针对性方法测试", test_targeted_approach),
        ("内容提取逻辑测试", test_content_extraction),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n开始测试: {test_name}")
            result = test_func()
            results[test_name] = result
            status = "通过" if result else "失败"
            logger.info(f"测试结果: {status}")
        except Exception as e:
            logger.error(f"测试异常: {e}")
            results[test_name] = False
    
    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！可以运行针对性爬虫")
        logger.info("\n运行命令:")
        logger.info("python mastodon_scraper_targeted.py")
    else:
        logger.warning("⚠️  部分测试失败，但针对性爬虫仍可尝试运行")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)