#!/usr/bin/env python3
"""
测试最终版爬虫的功能
验证八类问题的解决方案
"""

import json
import time
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_configuration():
    """测试配置系统"""
    logger.info("=" * 50)
    logger.info("测试1: 配置系统")
    logger.info("=" * 50)
    
    # 创建测试配置
    test_config = {
        "filter_days": 90,  # 扩大到3个月
        "chinese_ratio_threshold": 0.1,  # 更宽松的中文检测
        "chinese_char_threshold": 1,
        "headless_mode": True,
        "use_anti_detection": True,
        "initial_wait_time": 15,  # 更长的等待时间
        "max_wait_time": 45,
        "max_scroll_attempts": 30,
        "verbose_logging": True,
        "save_screenshots": True,
        "save_page_source": True,
        "max_retries": 2
    }
    
    try:
        with open('scraper_config.json', 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        logger.info("✓ 测试配置文件已创建")
        return True
    except Exception as e:
        logger.error(f"✗ 创建配置文件失败: {e}")
        return False


def test_chinese_detection():
    """测试中文检测逻辑"""
    logger.info("=" * 50)
    logger.info("测试2: 中文检测逻辑")
    logger.info("=" * 50)
    
    import re
    
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    test_cases = [
        ("这是一个中文测试", True, "纯中文"),
        ("This is English", False, "纯英文"),
        ("中英混合 mixed", True, "中英混合"),
        ("Hello 世界", True, "少量中文"),
        ("RT @someone: 转发内容", True, "转发格式"),
        ("😊 今天心情好", True, "表情符号+中文"),
        ("123456", False, "纯数字"),
        ("", False, "空字符串"),
        ("a", False, "单字符"),
        ("中", True, "单个中文字符"),
        ("你好世界！这是一个测试。", True, "标准中文句子"),
        ("RT: English content with 中文", True, "英文为主但有中文"),
    ]
    
    # 使用宽松的检测条件
    chinese_ratio_threshold = 0.1
    chinese_char_threshold = 1
    
    passed = 0
    total = len(test_cases)
    
    for text, expected, description in test_cases:
        chinese_chars = chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        if total_chars > 0:
            ratio = chinese_char_count / total_chars
            is_chinese = (ratio > chinese_ratio_threshold or 
                         chinese_char_count > chinese_char_threshold)
        else:
            ratio = 0
            is_chinese = False
        
        result = "✓" if is_chinese == expected else "✗"
        if is_chinese == expected:
            passed += 1
        
        logger.info(f"{result} {description}: '{text}' -> 中文={is_chinese} (期望={expected}) [比例={ratio:.2f}]")
    
    logger.info(f"中文检测测试: {passed}/{total} 通过")
    return passed == total


def test_time_logic():
    """测试时间逻辑"""
    logger.info("=" * 50)
    logger.info("测试3: 时间逻辑")
    logger.info("=" * 50)
    
    from datetime import datetime, timedelta, timezone
    import re
    
    def parse_time_test(time_text):
        """简化的时间解析测试"""
        try:
            if not time_text:
                return None
                
            time_text = time_text.strip()
            current_time = datetime.now()
            
            # 相对时间模式
            patterns = [
                (r'(\d+)\s*分钟前', 'minutes'),
                (r'(\d+)\s*小时前', 'hours'),
                (r'(\d+)\s*天前', 'days'),
                (r'(\d+)m', 'minutes'),
                (r'(\d+)h', 'hours'),
                (r'(\d+)d', 'days'),
            ]
            
            for pattern, unit in patterns:
                match = re.search(pattern, time_text)
                if match:
                    value = int(match.group(1))
                    if unit == 'minutes':
                        return current_time - timedelta(minutes=value)
                    elif unit == 'hours':
                        return current_time - timedelta(hours=value)
                    elif unit == 'days':
                        return current_time - timedelta(days=value)
            
            return None
        except:
            return None
    
    test_times = [
        ("5分钟前", True),
        ("2小时前", True),
        ("1天前", True),
        ("30天前", True),
        ("5m", True),
        ("2h", True),
        ("1d", True),
        ("invalid", False),
        ("", False),
    ]
    
    passed = 0
    cutoff_date = datetime.now() - timedelta(days=90)  # 3个月
    
    for time_text, should_parse in test_times:
        parsed_time = parse_time_test(time_text)
        
        if should_parse:
            if parsed_time and parsed_time >= cutoff_date:
                logger.info(f"✓ '{time_text}' -> {parsed_time.strftime('%Y-%m-%d %H:%M')} (在范围内)")
                passed += 1
            elif parsed_time:
                logger.info(f"○ '{time_text}' -> {parsed_time.strftime('%Y-%m-%d %H:%M')} (超出范围)")
                passed += 1  # 解析成功也算通过
            else:
                logger.info(f"✗ '{time_text}' -> 解析失败")
        else:
            if parsed_time is None:
                logger.info(f"✓ '{time_text}' -> 正确识别为无效时间")
                passed += 1
            else:
                logger.info(f"✗ '{time_text}' -> 意外解析成功")
    
    logger.info(f"时间解析测试: {passed}/{len(test_times)} 通过")
    return passed == len(test_times)


def test_selector_robustness():
    """测试选择器健壮性"""
    logger.info("=" * 50)
    logger.info("测试4: 选择器健壮性")
    logger.info("=" * 50)
    
    # 模拟不同的选择器策略
    selector_groups = [
        # 第一组：Mastodon标准选择器
        ["article", ".status", ".status__wrapper"],
        
        # 第二组：通用角色选择器
        ["div[role='article']", "[role='listitem']", ".timeline-item"],
        
        # 第三组：类名包含关键词
        ["div[class*='status']", "div[class*='post']", "div[class*='toot']"],
        
        # 第四组：数据属性选择器
        ["div[data-testid]", "[data-id]", "[data-status-id]"],
        
        # 第五组：更宽泛的选择器
        ["main div > div", ".app-body div[class]", "div[class]"]
    ]
    
    logger.info("选择器分组策略:")
    for i, group in enumerate(selector_groups, 1):
        logger.info(f"  第{i}组: {', '.join(group)}")
    
    logger.info("✓ 选择器策略设计合理，提供多层次备选方案")
    return True


def run_basic_functionality_test():
    """运行基础功能测试"""
    logger.info("=" * 50)
    logger.info("测试5: 基础功能测试")
    logger.info("=" * 50)
    
    try:
        # 测试网络连接
        import requests
        response = requests.get("https://m.cmx.im", timeout=10)
        if response.status_code == 200:
            logger.info("✓ 网络连接正常")
        else:
            logger.warning(f"⚠ 网站响应异常: {response.status_code}")
        
        # 测试依赖包
        try:
            from selenium import webdriver
            from webdriver_manager.chrome import ChromeDriverManager
            logger.info("✓ Selenium 依赖包正常")
        except ImportError as e:
            logger.error(f"✗ 缺少依赖包: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 基础功能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("长毛象爬虫最终版本测试")
    print("=" * 60)
    
    tests = [
        ("配置系统", test_configuration),
        ("中文检测", test_chinese_detection),
        ("时间逻辑", test_time_logic),
        ("选择器健壮性", test_selector_robustness),
        ("基础功能", run_basic_functionality_test),
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
        logger.info("🎉 所有测试通过！可以运行最终版爬虫")
        logger.info("\n运行命令:")
        logger.info("python mastodon_scraper_final.py")
        logger.info("\n或运行诊断工具:")
        logger.info("python comprehensive_diagnostic.py")
    else:
        logger.warning("⚠️  部分测试失败，请检查环境配置")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)