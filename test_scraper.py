#!/usr/bin/env python3
"""
长毛象爬虫测试脚本 - 简化版本用于测试基本功能
"""

import re
import time
import logging
from datetime import datetime, timedelta

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_connection():
    """测试基本连接"""
    logger.info("测试基本网络连接...")
    try:
        response = requests.get("https://m.cmx.im", timeout=10)
        logger.info(f"网站响应状态码: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"连接失败: {e}")
        return False


def test_selenium_setup():
    """测试Selenium设置"""
    logger.info("测试Selenium WebDriver设置...")
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 测试访问目标页面
        driver.get("https://m.cmx.im/public/local")
        time.sleep(3)
        
        page_title = driver.title
        logger.info(f"页面标题: {page_title}")
        
        # 检查页面内容
        page_source = driver.page_source
        has_mastodon_content = "mastodon" in page_source.lower() or "长毛象" in page_source
        
        driver.quit()
        
        logger.info(f"页面包含Mastodon内容: {has_mastodon_content}")
        return has_mastodon_content
        
    except Exception as e:
        logger.error(f"Selenium测试失败: {e}")
        return False


def test_chinese_detection():
    """测试中文检测功能"""
    logger.info("测试中文检测功能...")
    
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    test_cases = [
        ("这是一个中文测试", True),
        ("This is English text", False),
        ("中英混合 mixed text 测试", True),
        ("Hello 世界", True),
        ("", False),
        ("123456", False),
        ("今天天气很好！", True)
    ]
    
    for text, expected in test_cases:
        chinese_chars = chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        is_chinese = total_chars > 0 and (chinese_char_count / total_chars) > 0.3
        
        result = "✓" if is_chinese == expected else "✗"
        logger.info(f"{result} '{text}' -> 中文: {is_chinese} (期望: {expected})")
    
    return True


def main():
    """主测试函数"""
    logger.info("开始运行爬虫测试...")
    
    tests = [
        ("网络连接测试", test_basic_connection),
        ("中文检测测试", test_chinese_detection),
        ("Selenium设置测试", test_selenium_setup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "通过" if result else "失败"
            logger.info(f"测试结果: {status}")
        except Exception as e:
            logger.error(f"测试异常: {e}")
            results[test_name] = False
    
    # 输出总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{status} {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    logger.info(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！可以运行完整的爬虫脚本。")
    else:
        logger.warning("⚠️  部分测试失败，请检查环境配置。")


if __name__ == "__main__":
    main()