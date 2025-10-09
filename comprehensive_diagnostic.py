#!/usr/bin/env python3
"""
长毛象爬虫全面诊断系统
针对八类常见问题进行系统性检查和解决
"""

import re
import json
import time
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_diagnostic.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonDiagnosticTool:
    """长毛象爬虫全面诊断工具"""
    
    def __init__(self, base_url: str = "https://m.cmx.im"):
        self.base_url = base_url
        self.target_url = f"{base_url}/public/local"
        self.driver = None
        self.diagnostic_results = {}
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 时间过滤 - 近两个月
        self.cutoff_date = datetime.now() - timedelta(days=60)
        
    def setup_driver(self, headless: bool = True, anti_detection: bool = True) -> None:
        """设置Chrome WebDriver"""
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument('--headless')
            
            # 基础设置
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            if anti_detection:
                # 反检测设置
                chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                
                # 禁用图片和CSS加载以提高速度（可选）
                prefs = {
                    "profile.managed_default_content_settings.images": 1,  # 允许图片
                    "profile.default_content_setting_values.notifications": 2
                }
                chrome_options.add_experimental_option("prefs", prefs)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            if anti_detection:
                # 执行反检测脚本
                self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
                self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']})")
            
            logger.info(f"Chrome WebDriver 设置完成 (headless={headless}, anti_detection={anti_detection})")
            
        except Exception as e:
            logger.error(f"设置WebDriver失败: {e}")
            raise
    
    def test_1_login_status(self) -> Dict:
        """一、运行环境与登录状态检查"""
        logger.info("=" * 60)
        logger.info("测试1: 运行环境与登录状态检查")
        logger.info("=" * 60)
        
        results = {
            "test_name": "登录状态检查",
            "passed": False,
            "details": {},
            "recommendations": []
        }
        
        try:
            # 1.1 检查是否需要登录
            logger.info("1.1 检查页面访问权限...")
            self.driver.get(self.target_url)
            time.sleep(5)
            
            page_title = self.driver.title
            page_url = self.driver.current_url
            
            results["details"]["page_title"] = page_title
            results["details"]["current_url"] = page_url
            results["details"]["target_url"] = self.target_url
            
            # 检查是否被重定向到登录页面
            if "login" in page_url.lower() or "sign" in page_url.lower():
                results["details"]["login_required"] = True
                results["recommendations"].append("网站需要登录才能访问内容")
                logger.warning("⚠ 页面被重定向到登录页面")
            else:
                results["details"]["login_required"] = False
                logger.info("✓ 页面可以直接访问，无需登录")
            
            # 1.2 检查cookies和会话状态
            logger.info("1.2 检查cookies和会话状态...")
            cookies = self.driver.get_cookies()
            results["details"]["cookies_count"] = len(cookies)
            results["details"]["has_session_cookies"] = any(
                cookie.get("name", "").lower() in ["session", "sessionid", "_session", "connect.sid"]
                for cookie in cookies
            )
            
            logger.info(f"   找到 {len(cookies)} 个cookies")
            
            # 1.3 检查页面内容完整性
            logger.info("1.3 检查页面内容完整性...")
            page_source = self.driver.page_source
            
            # 检查是否包含登录提示
            login_indicators = ["登录", "sign in", "log in", "请先登录", "需要登录"]
            has_login_prompt = any(indicator in page_source.lower() for indicator in login_indicators)
            results["details"]["has_login_prompt"] = has_login_prompt
            
            # 检查是否包含内容加载元素
            content_indicators = ["timeline", "status", "post", "toot", "mastodon"]
            has_content_indicators = any(indicator in page_source.lower() for indicator in content_indicators)
            results["details"]["has_content_indicators"] = has_content_indicators
            
            if has_login_prompt:
                logger.warning("⚠ 页面包含登录提示")
                results["recommendations"].append("页面要求登录，需要实现自动登录功能")
            
            if has_content_indicators:
                logger.info("✓ 页面包含内容相关元素")
                results["passed"] = True
            else:
                logger.warning("⚠ 页面缺少内容相关元素")
                results["recommendations"].append("页面可能未正确加载或需要特殊权限")
            
            # 1.4 保存页面截图和源码
            try:
                self.driver.save_screenshot("diagnostic_login_check.png")
                with open("diagnostic_login_page_source.html", "w", encoding="utf-8") as f:
                    f.write(page_source)
                logger.info("✓ 页面截图和源码已保存")
            except Exception as e:
                logger.warning(f"保存截图或源码失败: {e}")
            
        except Exception as e:
            logger.error(f"登录状态检查失败: {e}")
            results["recommendations"].append(f"检查过程出错: {e}")
        
        self.diagnostic_results["login_status"] = results
        return results
    
    def test_2_page_loading(self) -> Dict:
        """二、页面显示与加载逻辑检查"""
        logger.info("=" * 60)
        logger.info("测试2: 页面显示与加载逻辑检查")
        logger.info("=" * 60)
        
        results = {
            "test_name": "页面加载检查",
            "passed": False,
            "details": {},
            "recommendations": []
        }
        
        try:
            # 2.1 检查页面初始状态
            logger.info("2.1 检查页面初始状态...")
            initial_height = self.driver.execute_script("return document.body.scrollHeight")
            initial_elements = len(self.driver.find_elements(By.TAG_NAME, "div"))
            
            results["details"]["initial_height"] = initial_height
            results["details"]["initial_elements"] = initial_elements
            
            # 2.2 等待JavaScript加载
            logger.info("2.2 等待JavaScript内容加载...")
            wait_times = [5, 10, 15, 20]
            loading_progress = {}
            
            for wait_time in wait_times:
                time.sleep(5)  # 每次等待5秒
                current_height = self.driver.execute_script("return document.body.scrollHeight")
                current_elements = len(self.driver.find_elements(By.TAG_NAME, "div"))
                
                loading_progress[f"{wait_time}s"] = {
                    "height": current_height,
                    "elements": current_elements,
                    "height_changed": current_height != initial_height,
                    "elements_changed": current_elements != initial_elements
                }
                
                logger.info(f"   {wait_time}秒后: 高度={current_height}, 元素={current_elements}")
            
            results["details"]["loading_progress"] = loading_progress
            
            # 2.3 检查动态内容加载
            logger.info("2.3 检查动态内容加载...")
            
            # 尝试滚动触发内容加载
            scroll_results = {}
            for i in range(3):
                before_scroll = self.driver.execute_script("return document.body.scrollHeight")
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                after_scroll = self.driver.execute_script("return document.body.scrollHeight")
                
                scroll_results[f"scroll_{i+1}"] = {
                    "before": before_scroll,
                    "after": after_scroll,
                    "changed": after_scroll != before_scroll
                }
                
                logger.info(f"   滚动{i+1}: {before_scroll} -> {after_scroll}")
            
            results["details"]["scroll_results"] = scroll_results
            
            # 2.4 检查是否有内容元素出现
            logger.info("2.4 检查内容元素...")
            
            content_selectors = [
                "article", ".status", "[role='article']", 
                "div[class*='status']", "[data-testid]",
                ".timeline-item", "div[class*='post']", "div[class*='toot']"
            ]
            
            found_elements = {}
            total_found = 0
            
            for selector in content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    found_elements[selector] = len(elements)
                    total_found += len(elements)
                    if elements:
                        logger.info(f"   ✓ {selector}: {len(elements)} 个元素")
                except Exception as e:
                    found_elements[selector] = 0
                    logger.debug(f"   ✗ {selector}: 查找失败 - {e}")
            
            results["details"]["found_elements"] = found_elements
            results["details"]["total_elements_found"] = total_found
            
            if total_found > 0:
                results["passed"] = True
                logger.info(f"✓ 找到 {total_found} 个潜在内容元素")
            else:
                logger.warning("⚠ 未找到任何内容元素")
                results["recommendations"].append("页面可能需要更长的加载时间或特殊触发条件")
            
            # 2.5 保存最终状态
            try:
                self.driver.save_screenshot("diagnostic_page_loading.png")
                logger.info("✓ 页面加载状态截图已保存")
            except Exception as e:
                logger.warning(f"保存截图失败: {e}")
            
        except Exception as e:
            logger.error(f"页面加载检查失败: {e}")
            results["recommendations"].append(f"检查过程出错: {e}")
        
        self.diagnostic_results["page_loading"] = results
        return results
    
    def test_3_filter_conditions(self) -> Dict:
        """三、筛选条件与时间逻辑检查"""
        logger.info("=" * 60)
        logger.info("测试3: 筛选条件与时间逻辑检查")
        logger.info("=" * 60)
        
        results = {
            "test_name": "筛选条件检查",
            "passed": False,
            "details": {},
            "recommendations": []
        }
        
        try:
            # 3.1 时间逻辑检查
            logger.info("3.1 检查时间逻辑...")
            
            current_time = datetime.now()
            current_utc = datetime.now(timezone.utc)
            cutoff_local = current_time - timedelta(days=60)
            cutoff_utc = current_utc - timedelta(days=60)
            
            time_info = {
                "current_local": current_time.isoformat(),
                "current_utc": current_utc.isoformat(),
                "cutoff_local": cutoff_local.isoformat(),
                "cutoff_utc": cutoff_utc.isoformat(),
                "timezone_offset": current_time.utcoffset(),
                "filter_days": 60
            }
            
            results["details"]["time_logic"] = time_info
            logger.info(f"   当前本地时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   当前UTC时间: {current_utc.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   过滤截止时间: {cutoff_local.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 3.2 中文检测逻辑检查
            logger.info("3.2 检查中文检测逻辑...")
            
            test_texts = [
                "这是一个纯中文测试文本",
                "This is pure English text",
                "中英混合 mixed content 测试",
                "Hello 世界！",
                "今天天气很好 😊",
                "RT @someone: 转发的中文内容",
                "短文本",
                "a",
                "",
                "123456789",
                "🎉🎊🎈",
                "中文内容但是很长很长很长很长很长很长很长很长很长很长很长很长English content"
            ]
            
            chinese_detection_results = {}
            
            for text in test_texts:
                chinese_chars = self.chinese_pattern.findall(text)
                chinese_char_count = sum(len(chars) for chars in chinese_chars)
                total_chars = len(text.strip())
                
                if total_chars > 0:
                    ratio = chinese_char_count / total_chars
                    is_chinese_strict = ratio > 0.3  # 严格条件
                    is_chinese_loose = ratio > 0.15 or chinese_char_count > 2  # 宽松条件
                else:
                    ratio = 0
                    is_chinese_strict = False
                    is_chinese_loose = False
                
                chinese_detection_results[text[:30]] = {
                    "total_chars": total_chars,
                    "chinese_chars": chinese_char_count,
                    "ratio": ratio,
                    "is_chinese_strict": is_chinese_strict,
                    "is_chinese_loose": is_chinese_loose
                }
                
                logger.info(f"   '{text[:30]}': 中文比例={ratio:.2f}, 严格={is_chinese_strict}, 宽松={is_chinese_loose}")
            
            results["details"]["chinese_detection"] = chinese_detection_results
            
            # 3.3 筛选条件建议
            logger.info("3.3 生成筛选条件建议...")
            
            # 统计通过不同条件的文本数量
            strict_pass = sum(1 for r in chinese_detection_results.values() if r["is_chinese_strict"])
            loose_pass = sum(1 for r in chinese_detection_results.values() if r["is_chinese_loose"])
            
            results["details"]["filter_stats"] = {
                "total_test_texts": len(test_texts),
                "strict_condition_pass": strict_pass,
                "loose_condition_pass": loose_pass
            }
            
            if loose_pass > strict_pass:
                results["recommendations"].append("建议使用宽松的中文检测条件（比例>15%或中文字符>2个）")
            
            results["recommendations"].append("考虑扩大时间范围到90天或更长")
            results["recommendations"].append("可以先不设置时间过滤，确认是否有任何中文内容")
            
            results["passed"] = True
            
        except Exception as e:
            logger.error(f"筛选条件检查失败: {e}")
            results["recommendations"].append(f"检查过程出错: {e}")
        
        self.diagnostic_results["filter_conditions"] = results
        return results
    
    def test_4_page_structure(self) -> Dict:
        """四、选择器与页面结构检查"""
        logger.info("=" * 60)
        logger.info("测试4: 选择器与页面结构检查")
        logger.info("=" * 60)
        
        results = {
            "test_name": "页面结构检查",
            "passed": False,
            "details": {},
            "recommendations": []
        }
        
        try:
            # 4.1 分析页面DOM结构
            logger.info("4.1 分析页面DOM结构...")
            
            # 获取页面基本信息
            page_info = {
                "title": self.driver.title,
                "url": self.driver.current_url,
                "total_elements": len(self.driver.find_elements(By.TAG_NAME, "*")),
                "div_elements": len(self.driver.find_elements(By.TAG_NAME, "div")),
                "article_elements": len(self.driver.find_elements(By.TAG_NAME, "article")),
                "section_elements": len(self.driver.find_elements(By.TAG_NAME, "section"))
            }
            
            results["details"]["page_info"] = page_info
            logger.info(f"   页面标题: {page_info['title']}")
            logger.info(f"   总元素数: {page_info['total_elements']}")
            logger.info(f"   DIV元素数: {page_info['div_elements']}")
            
            # 4.2 测试各种选择器
            logger.info("4.2 测试各种CSS选择器...")
            
            selectors_to_test = [
                # Mastodon标准选择器
                ("article", "标准文章元素"),
                (".status", "状态类"),
                (".status__wrapper", "状态包装器"),
                ("[data-testid='status']", "测试ID状态"),
                (".detailed-status", "详细状态"),
                
                # 通用选择器
                ("div[role='article']", "角色为文章的DIV"),
                ("[role='listitem']", "列表项角色"),
                (".timeline-item", "时间线项"),
                
                # 类名包含关键词
                ("div[class*='status']", "类名包含status"),
                ("div[class*='post']", "类名包含post"),
                ("div[class*='toot']", "类名包含toot"),
                ("div[class*='timeline']", "类名包含timeline"),
                
                # 数据属性
                ("div[data-testid]", "有测试ID的DIV"),
                ("[data-id]", "有ID数据属性"),
                ("[data-status-id]", "有状态ID数据属性"),
                
                # 更宽泛的选择器
                ("main div", "主要区域的DIV"),
                (".app-body div", "应用主体的DIV"),
                ("div[class]", "有类名的DIV"),
                
                # 时间相关
                ("time", "时间元素"),
                (".timestamp", "时间戳类"),
                ("div[class*='time']", "类名包含time"),
                
                # 用户相关
                (".display-name", "显示名称"),
                (".username", "用户名"),
                ("div[class*='user']", "类名包含user"),
                ("div[class*='account']", "类名包含account"),
                
                # 内容相关
                (".content", "内容类"),
                (".text", "文本类"),
                ("div[class*='content']", "类名包含content"),
                ("p", "段落元素")
            ]
            
            selector_results = {}
            working_selectors = []
            
            for selector, description in selectors_to_test:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    count = len(elements)
                    
                    # 检查元素是否有文本内容
                    text_elements = 0
                    sample_texts = []
                    
                    for i, element in enumerate(elements[:5]):  # 只检查前5个
                        try:
                            text = element.text.strip()
                            if text and len(text) > 10:
                                text_elements += 1
                                if len(sample_texts) < 2:
                                    sample_texts.append(text[:100])
                        except:
                            pass
                    
                    selector_results[selector] = {
                        "description": description,
                        "count": count,
                        "text_elements": text_elements,
                        "sample_texts": sample_texts,
                        "effectiveness": text_elements / max(count, 1)
                    }
                    
                    if count > 0 and text_elements > 0:
                        working_selectors.append((selector, description, count, text_elements))
                        logger.info(f"   ✓ {selector} ({description}): {count}个元素, {text_elements}个有文本")
                    elif count > 0:
                        logger.info(f"   ○ {selector} ({description}): {count}个元素, 但无文本内容")
                    else:
                        logger.debug(f"   ✗ {selector} ({description}): 未找到元素")
                        
                except Exception as e:
                    selector_results[selector] = {
                        "description": description,
                        "error": str(e),
                        "count": 0,
                        "text_elements": 0
                    }
                    logger.debug(f"   ✗ {selector} ({description}): 查找失败 - {e}")
            
            results["details"]["selector_results"] = selector_results
            results["details"]["working_selectors"] = working_selectors
            
            # 4.3 推荐最佳选择器
            logger.info("4.3 推荐最佳选择器...")
            
            if working_selectors:
                # 按效果排序
                working_selectors.sort(key=lambda x: x[3], reverse=True)  # 按有文本的元素数排序
                best_selectors = working_selectors[:5]
                
                results["details"]["recommended_selectors"] = best_selectors
                results["passed"] = True
                
                logger.info("   推荐的选择器（按效果排序）:")
                for i, (selector, desc, total, text_count) in enumerate(best_selectors, 1):
                    logger.info(f"   {i}. {selector} - {text_count}个有效元素")
                    results["recommendations"].append(f"使用选择器: {selector} ({desc})")
            else:
                logger.warning("⚠ 未找到任何有效的选择器")
                results["recommendations"].append("页面结构可能与预期不符，需要手动分析DOM结构")
                results["recommendations"].append("建议保存页面源码进行详细分析")
            
            # 4.4 保存页面结构分析
            try:
                with open("diagnostic_page_structure.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                
                # 保存DOM结构摘要
                dom_summary = {
                    "page_info": page_info,
                    "working_selectors": working_selectors,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open("diagnostic_dom_summary.json", "w", encoding="utf-8") as f:
                    json.dump(dom_summary, f, ensure_ascii=False, indent=2)
                
                logger.info("✓ 页面结构分析文件已保存")
                
            except Exception as e:
                logger.warning(f"保存页面结构分析失败: {e}")
            
        except Exception as e:
            logger.error(f"页面结构检查失败: {e}")
            results["recommendations"].append(f"检查过程出错: {e}")
        
        self.diagnostic_results["page_structure"] = results
        return results
    
    def test_5_anti_bot_detection(self) -> Dict:
        """五、反爬虫与访问差异检查"""
        logger.info("=" * 60)
        logger.info("测试5: 反爬虫与访问差异检查")
        logger.info("=" * 60)
        
        results = {
            "test_name": "反爬虫检查",
            "passed": False,
            "details": {},
            "recommendations": []
        }
        
        try:
            # 5.1 检查User-Agent影响
            logger.info("5.1 检查User-Agent影响...")
            
            current_ua = self.driver.execute_script("return navigator.userAgent")
            results["details"]["current_user_agent"] = current_ua
            logger.info(f"   当前User-Agent: {current_ua[:100]}...")
            
            # 5.2 检查WebDriver检测
            logger.info("5.2 检查WebDriver检测...")
            
            webdriver_tests = {
                "webdriver_property": self.driver.execute_script("return navigator.webdriver"),
                "automation_property": self.driver.execute_script("return window.chrome && window.chrome.runtime && window.chrome.runtime.onConnect"),
                "plugins_count": self.driver.execute_script("return navigator.plugins.length"),
                "languages": self.driver.execute_script("return navigator.languages"),
                "platform": self.driver.execute_script("return navigator.platform"),
                "hardwareConcurrency": self.driver.execute_script("return navigator.hardwareConcurrency")
            }
            
            results["details"]["webdriver_detection"] = webdriver_tests
            
            for test_name, result in webdriver_tests.items():
                logger.info(f"   {test_name}: {result}")
            
            # 检查是否有明显的自动化标识
            automation_detected = (
                webdriver_tests["webdriver_property"] is not None or
                webdriver_tests["plugins_count"] == 0
            )
            
            results["details"]["automation_detected"] = automation_detected
            
            if automation_detected:
                logger.warning("⚠ 检测到自动化标识，可能被网站识别为爬虫")
                results["recommendations"].append("使用更强的反检测设置")
            else:
                logger.info("✓ 未检测到明显的自动化标识")
            
            # 5.3 检查页面内容差异
            logger.info("5.3 检查页面内容差异...")
            
            # 获取当前页面的关键指标
            page_metrics = {
                "page_size": len(self.driver.page_source),
                "title": self.driver.title,
                "url": self.driver.current_url,
                "has_javascript_errors": False  # 简化处理
            }
            
            # 检查是否有错误页面的标识
            error_indicators = ["error", "blocked", "forbidden", "access denied", "bot detected"]
            page_content_lower = self.driver.page_source.lower()
            
            detected_errors = [indicator for indicator in error_indicators if indicator in page_content_lower]
            
            results["details"]["page_metrics"] = page_metrics
            results["details"]["detected_errors"] = detected_errors
            
            if detected_errors:
                logger.warning(f"⚠ 检测到错误指示器: {detected_errors}")
                results["recommendations"].append("页面可能被反爬虫机制拦截")
            else:
                logger.info("✓ 未检测到错误页面标识")
            
            # 5.4 检查域名和子域差异
            logger.info("5.4 检查域名访问...")
            
            current_domain = self.driver.current_url
            results["details"]["accessed_domain"] = current_domain
            
            # 检查是否被重定向到其他域名
            if self.base_url not in current_domain:
                logger.warning(f"⚠ 页面被重定向: {self.base_url} -> {current_domain}")
                results["recommendations"].append("检查是否被重定向到其他域名或子域")
            else:
                logger.info("✓ 域名访问正常")
            
            # 5.5 综合评估
            if not automation_detected and not detected_errors:
                results["passed"] = True
                logger.info("✓ 反爬虫检查通过")
            else:
                logger.warning("⚠ 可能存在反爬虫限制")
                results["recommendations"].append("考虑使用更隐蔽的爬虫设置")
                results["recommendations"].append("尝试使用代理IP或更换User-Agent")
            
        except Exception as e:
            logger.error(f"反爬虫检查失败: {e}")
            results["recommendations"].append(f"检查过程出错: {e}")
        
        self.diagnostic_results["anti_bot_detection"] = results
        return results
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合诊断报告"""
        logger.info("=" * 60)
        logger.info("生成综合诊断报告")
        logger.info("=" * 60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "target_url": self.target_url,
            "test_results": self.diagnostic_results,
            "overall_assessment": {},
            "recommended_actions": []
        }
        
        # 统计测试通过情况
        passed_tests = sum(1 for result in self.diagnostic_results.values() if result.get("passed", False))
        total_tests = len(self.diagnostic_results)
        
        report["overall_assessment"] = {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASS" if passed_tests >= total_tests * 0.6 else "FAIL"
        }
        
        # 收集所有建议
        all_recommendations = []
        for test_result in self.diagnostic_results.values():
            all_recommendations.extend(test_result.get("recommendations", []))
        
        report["recommended_actions"] = list(set(all_recommendations))  # 去重
        
        # 输出报告摘要
        logger.info(f"诊断完成: {passed_tests}/{total_tests} 项测试通过")
        logger.info(f"整体状态: {report['overall_assessment']['overall_status']}")
        
        if report["recommended_actions"]:
            logger.info("主要建议:")
            for i, action in enumerate(report["recommended_actions"][:5], 1):
                logger.info(f"  {i}. {action}")
        
        # 保存报告
        try:
            with open("comprehensive_diagnostic_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("✓ 综合诊断报告已保存到 comprehensive_diagnostic_report.json")
        except Exception as e:
            logger.warning(f"保存报告失败: {e}")
        
        return report
    
    def run_all_tests(self, headless: bool = False) -> Dict:
        """运行所有诊断测试"""
        logger.info("开始运行长毛象爬虫全面诊断...")
        logger.info(f"目标URL: {self.target_url}")
        logger.info(f"无头模式: {headless}")
        
        try:
            # 设置WebDriver
            self.setup_driver(headless=headless, anti_detection=True)
            
            # 运行所有测试
            self.test_1_login_status()
            self.test_2_page_loading()
            self.test_3_filter_conditions()
            self.test_4_page_structure()
            self.test_5_anti_bot_detection()
            
            # 生成综合报告
            report = self.generate_comprehensive_report()
            
            return report
            
        except Exception as e:
            logger.error(f"诊断过程失败: {e}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver 已关闭")


def main():
    """主函数"""
    print("长毛象爬虫全面诊断工具")
    print("=" * 60)
    
    # 询问是否使用无头模式
    use_headless = input("是否使用无头模式？(y/N): ").lower().strip() == 'y'
    
    diagnostic_tool = MastodonDiagnosticTool()
    
    try:
        report = diagnostic_tool.run_all_tests(headless=use_headless)
        
        print("\n" + "=" * 60)
        print("诊断完成！")
        print(f"整体状态: {report['overall_assessment']['overall_status']}")
        print(f"通过率: {report['overall_assessment']['success_rate']:.1%}")
        print("\n生成的文件:")
        print("- comprehensive_diagnostic_report.json (详细报告)")
        print("- comprehensive_diagnostic.log (运行日志)")
        print("- diagnostic_*.png (页面截图)")
        print("- diagnostic_*.html (页面源码)")
        
    except Exception as e:
        print(f"诊断失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())