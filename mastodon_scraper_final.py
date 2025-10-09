#!/usr/bin/env python3
"""
长毛象爬虫最终版本 - 基于全面诊断的综合解决方案
解决八类常见问题的完整爬虫实现
"""

import re
import json
import csv
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
        logging.FileHandler('mastodon_scraper_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonScraperFinal:
    """长毛象爬虫最终版本 - 综合解决方案"""
    
    def __init__(self, base_url: str = "https://m.cmx.im"):
        self.base_url = base_url
        self.target_url = f"{base_url}/public/local"
        self.driver = None
        self.posts_data = []
        self.config = self.load_config()
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 时间过滤配置
        self.cutoff_date = datetime.now() - timedelta(days=self.config['filter_days'])
        
    def load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            # 时间过滤设置
            'filter_days': 60,
            'use_utc_time': False,
            
            # 中文检测设置
            'chinese_ratio_threshold': 0.15,  # 宽松设置
            'chinese_char_threshold': 2,      # 宽松设置
            'min_content_length': 3,          # 最小内容长度
            
            # 页面加载设置
            'initial_wait_time': 10,          # 初始等待时间
            'max_wait_time': 30,              # 最大等待时间
            'scroll_wait_time': 3,            # 滚动等待时间
            'max_scroll_attempts': 20,        # 最大滚动次数
            
            # 反爬虫设置
            'use_anti_detection': True,
            'headless_mode': True,
            'custom_user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # 重试设置
            'max_retries': 3,
            'retry_delay': 5,
            
            # 调试设置
            'save_screenshots': True,
            'save_page_source': True,
            'verbose_logging': True
        }
        
        # 尝试从文件加载配置
        try:
            with open('scraper_config.json', 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                logger.info("✓ 已加载用户配置文件")
        except FileNotFoundError:
            logger.info("使用默认配置")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return default_config
    
    def setup_driver(self) -> None:
        """设置Chrome WebDriver - 最强反检测版本"""
        try:
            chrome_options = Options()
            
            # 基础设置
            if self.config['headless_mode']:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            
            if self.config['use_anti_detection']:
                # 最强反检测设置
                chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument(f'--user-agent={self.config["custom_user_agent"]}')
                
                # 额外的反检测设置
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins-discovery')
                chrome_options.add_argument('--disable-web-security')
                chrome_options.add_argument('--allow-running-insecure-content')
                chrome_options.add_argument('--no-first-run')
                chrome_options.add_argument('--disable-default-apps')
                
                # 设置首选项
                prefs = {
                    "profile.managed_default_content_settings.images": 1,
                    "profile.default_content_setting_values.notifications": 2,
                    "profile.default_content_settings.popups": 0,
                    "profile.managed_default_content_settings.media_stream": 2,
                }
                chrome_options.add_experimental_option("prefs", prefs)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            if self.config['use_anti_detection']:
                # 执行反检测脚本
                self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
                self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN', 'zh', 'en']})")
                self.driver.execute_script("window.navigator.chrome = {runtime: {}};")
                self.driver.execute_script("Object.defineProperty(navigator, 'permissions', {get: () => ({query: () => Promise.resolve({state: 'granted'})})});")
            
            logger.info("Chrome WebDriver 设置完成（最强反检测模式）")
            
        except Exception as e:
            logger.error(f"设置WebDriver失败: {e}")
            raise
    
    def check_login_status(self) -> bool:
        """检查登录状态"""
        try:
            logger.info("检查登录状态...")
            
            # 检查是否被重定向到登录页面
            current_url = self.driver.current_url
            if "login" in current_url.lower() or "sign" in current_url.lower():
                logger.warning("页面被重定向到登录页面")
                return False
            
            # 检查页面内容是否包含登录提示
            page_source = self.driver.page_source.lower()
            login_indicators = ["请先登录", "需要登录", "sign in", "log in"]
            
            if any(indicator in page_source for indicator in login_indicators):
                logger.warning("页面包含登录提示")
                return False
            
            logger.info("✓ 无需登录或已登录")
            return True
            
        except Exception as e:
            logger.warning(f"检查登录状态失败: {e}")
            return True  # 默认假设不需要登录
    
    def wait_for_content_load(self) -> bool:
        """智能等待内容加载"""
        logger.info("等待页面内容加载...")
        
        try:
            # 等待应用容器
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "mastodon"))
            )
            logger.info("✓ Mastodon应用容器已加载")
            
            # 初始等待
            time.sleep(self.config['initial_wait_time'])
            
            # 动态等待内容出现
            max_wait = self.config['max_wait_time']
            wait_interval = 3
            waited_time = 0
            
            content_selectors = [
                "article", ".status", "[role='article']", 
                "div[class*='status']", "[data-testid]"
            ]
            
            while waited_time < max_wait:
                for selector in content_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            # 检查是否有文本内容
                            text_elements = [el for el in elements[:5] if el.text.strip()]
                            if text_elements:
                                logger.info(f"✓ 找到内容元素: {selector} ({len(text_elements)}个有文本)")
                                return True
                    except:
                        continue
                
                # 尝试滚动触发加载
                if waited_time % 9 == 0:  # 每9秒滚动一次
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    logger.info("尝试滚动触发内容加载...")
                
                time.sleep(wait_interval)
                waited_time += wait_interval
                
                if waited_time % 10 == 0:
                    logger.info(f"已等待 {waited_time}秒...")
            
            logger.warning(f"等待 {max_wait}秒后仍未找到内容")
            return False
            
        except TimeoutException:
            logger.error("页面加载超时")
            return False
        except Exception as e:
            logger.error(f"等待内容加载失败: {e}")
            return False
    
    def is_chinese_content(self, text: str) -> bool:
        """改进的中文内容检测"""
        if not text or len(text.strip()) < self.config['min_content_length']:
            return False
        
        chinese_chars = self.chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        ratio = chinese_char_count / total_chars
        
        # 使用配置的阈值
        return (ratio > self.config['chinese_ratio_threshold'] or 
                chinese_char_count > self.config['chinese_char_threshold'])
    
    def parse_time_with_timezone(self, time_text: str) -> Optional[datetime]:
        """改进的时间解析，考虑时区"""
        try:
            if not time_text:
                return None
                
            time_text = time_text.strip()
            
            # 获取当前时间（根据配置选择UTC或本地时间）
            if self.config['use_utc_time']:
                current_time = datetime.now(timezone.utc)
            else:
                current_time = datetime.now()
            
            # 处理相对时间
            relative_patterns = [
                (r'(\d+)\s*分钟前', 'minutes'),
                (r'(\d+)\s*小时前', 'hours'),
                (r'(\d+)\s*天前', 'days'),
                (r'(\d+)\s*周前', 'weeks'),
                (r'(\d+)\s*月前', 'months'),
                (r'(\d+)\s*年前', 'years'),
                (r'(\d+)m', 'minutes'),
                (r'(\d+)h', 'hours'),
                (r'(\d+)d', 'days'),
                (r'(\d+)w', 'weeks'),
            ]
            
            for pattern, unit in relative_patterns:
                match = re.search(pattern, time_text)
                if match:
                    value = int(match.group(1))
                    
                    if unit == 'minutes':
                        return current_time - timedelta(minutes=value)
                    elif unit == 'hours':
                        return current_time - timedelta(hours=value)
                    elif unit == 'days':
                        return current_time - timedelta(days=value)
                    elif unit == 'weeks':
                        return current_time - timedelta(weeks=value)
                    elif unit == 'months':
                        return current_time - timedelta(days=value*30)
                    elif unit == 'years':
                        return current_time - timedelta(days=value*365)
            
            # 尝试解析绝对时间格式
            absolute_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
                '%m-%d %H:%M',
                '%H:%M'
            ]
            
            for fmt in absolute_formats:
                try:
                    return datetime.strptime(time_text, fmt)
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.debug(f"解析时间失败: {time_text}, 错误: {e}")
        
        return None
    
    def find_post_elements_comprehensive(self) -> List:
        """全面的帖子元素查找"""
        logger.info("使用全面策略查找帖子元素...")
        
        # 按优先级排序的选择器列表
        selector_groups = [
            # 第一组：Mastodon标准选择器
            [
                "article",
                ".status",
                ".status__wrapper",
                "[data-testid='status']",
                ".detailed-status"
            ],
            
            # 第二组：通用角色选择器
            [
                "div[role='article']",
                "[role='listitem']",
                ".timeline-item",
                "main article",
                "section article"
            ],
            
            # 第三组：类名包含关键词
            [
                "div[class*='status']",
                "div[class*='post']",
                "div[class*='toot']",
                "div[class*='timeline']",
                "div[class*='item']"
            ],
            
            # 第四组：数据属性选择器
            [
                "div[data-testid]",
                "[data-id]",
                "[data-status-id]",
                "div[data-react-class]"
            ],
            
            # 第五组：更宽泛的选择器
            [
                "main div > div",
                ".app-body div[class]",
                "div[class]:has(time)",
                "div:has(.timestamp)"
            ]
        ]
        
        for group_index, selectors in enumerate(selector_groups, 1):
            logger.info(f"尝试第{group_index}组选择器...")
            
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if not elements:
                        continue
                    
                    # 过滤有效元素
                    valid_elements = []
                    for element in elements:
                        try:
                            text = element.text.strip()
                            if text and len(text) >= self.config['min_content_length']:
                                valid_elements.append(element)
                        except:
                            continue
                    
                    if valid_elements:
                        logger.info(f"✓ 使用选择器 '{selector}' 找到 {len(valid_elements)} 个有效元素")
                        return valid_elements
                    else:
                        logger.debug(f"选择器 '{selector}' 找到 {len(elements)} 个元素，但无有效文本")
                        
                except Exception as e:
                    logger.debug(f"选择器 '{selector}' 查找失败: {e}")
        
        logger.warning("所有选择器都未找到有效元素")
        return []
    
    def extract_post_comprehensive(self, post_element) -> Optional[Dict]:
        """全面的帖子信息提取"""
        try:
            # 获取元素的所有文本
            full_text = post_element.text.strip()
            if not full_text or len(full_text) < self.config['min_content_length']:
                return None
            
            post_data = {}
            
            # 1. 提取用户名 - 多种策略
            username_selectors = [
                ".display-name__account", ".status__display-name", "[data-testid='username']",
                ".account__display-name", ".username", ".author", ".user-name",
                "span[class*='username']", "div[class*='username']", "a[class*='username']",
                "span[class*='display-name']", "div[class*='display-name']",
                "a[href*='/@']", ".account-link", ".profile-link"
            ]
            
            username = None
            for selector in username_selectors:
                try:
                    username_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    username_text = username_element.text.strip()
                    if username_text:
                        username = username_text
                        break
                except:
                    continue
            
            # 从链接中提取用户名
            if not username:
                try:
                    links = post_element.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href")
                        if href and "/@" in href:
                            username_part = href.split("/@")[-1].split("/")[0]
                            if username_part:
                                username = f"@{username_part}"
                                break
                except:
                    pass
            
            if not username:
                username = "未知用户"
            
            # 2. 提取内容 - 多种策略
            content_selectors = [
                ".status__content", ".status__content__text", "[data-testid='statusContent']",
                ".detailed-status__content", ".status-content", ".content", ".post-content",
                ".text", ".message", ".body", "div[class*='content']", "p"
            ]
            
            content = None
            for selector in content_selectors:
                try:
                    content_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    content_text = content_element.text.strip()
                    if content_text and len(content_text) >= self.config['min_content_length']:
                        content = content_text
                        break
                except:
                    continue
            
            # 如果没找到特定内容元素，使用整个元素的文本
            if not content:
                content = full_text
            
            # 清理内容
            if content and username != "未知用户":
                clean_username = username.replace("@", "")
                content = content.replace(username, "").replace(clean_username, "").strip()
            
            # 移除常见的非内容文本
            noise_patterns = [
                r'^\d+\s*(分钟|小时|天|周|月|年)前',
                r'^\d+[mhdwy]',
                r'^RT\s*@\w+:?',
                r'转发了$',
                r'点赞$',
                r'评论$',
                r'分享$'
            ]
            
            for pattern in noise_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE).strip()
            
            # 3. 提取时间
            time_selectors = [
                "time", ".status__relative-time", "[data-testid='timestamp']",
                ".detailed-status__datetime", ".timestamp", ".time", ".date",
                "span[class*='time']", "div[class*='time']", "a[class*='time']",
                ".created-at", ".published"
            ]
            
            time_text = None
            post_url = None
            
            for selector in time_selectors:
                try:
                    time_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    time_text = time_element.text.strip()
                    
                    # 尝试获取链接
                    if time_element.tag_name == 'a':
                        post_url = time_element.get_attribute('href')
                    else:
                        try:
                            parent_link = time_element.find_element(By.XPATH, ".//ancestor::a[1]")
                            post_url = parent_link.get_attribute('href')
                        except:
                            pass
                    
                    if time_text:
                        break
                except:
                    continue
            
            # 4. 验证内容
            if not content or len(content.strip()) < self.config['min_content_length']:
                return None
            
            # 检查是否为中文内容
            if not self.is_chinese_content(content):
                return None
            
            # 5. 时间过滤
            post_time = self.parse_time_with_timezone(time_text) if time_text else None
            if post_time and post_time < self.cutoff_date:
                return None
            
            # 6. 构建帖子URL
            if post_url and not post_url.startswith('http'):
                post_url = urljoin(self.base_url, post_url)
            
            post_data = {
                'username': username,
                'content': content,
                'time': time_text or "未知时间",
                'parsed_time': post_time.isoformat() if post_time else None,
                'url': post_url or self.target_url,
                'content_length': len(content),
                'chinese_ratio': len(self.chinese_pattern.findall(content)) / len(content) if content else 0
            }
            
            return post_data
            
        except Exception as e:
            logger.debug(f"提取帖子信息失败: {e}")
            return None
    
    def scroll_and_load_comprehensive(self) -> None:
        """全面的滚动加载策略"""
        logger.info("开始智能滚动加载...")
        
        max_scrolls = self.config['max_scroll_attempts']
        scroll_wait = self.config['scroll_wait_time']
        
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        no_change_count = 0
        scroll_count = 0
        
        while scroll_count < max_scrolls and no_change_count < 3:
            # 多种滚动策略
            if scroll_count % 3 == 0:
                # 滚动到底部
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            elif scroll_count % 3 == 1:
                # 滚动到中间位置
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            else:
                # 平滑滚动
                self.driver.execute_script("window.scrollBy(0, 500);")
            
            time.sleep(scroll_wait)
            
            # 检查页面变化
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                no_change_count += 1
                logger.debug(f"页面高度未变化 ({no_change_count}/3)")
            else:
                no_change_count = 0
                logger.info(f"页面高度变化: {last_height} -> {new_height}")
            
            last_height = new_height
            scroll_count += 1
            
            # 定期检查是否有新内容
            if scroll_count % 5 == 0:
                elements = self.driver.find_elements(By.CSS_SELECTOR, "div")
                text_elements = len([el for el in elements if el.text.strip()])
                logger.info(f"滚动 {scroll_count} 次，当前有文本元素: {text_elements}")
        
        logger.info(f"滚动完成，总共滚动 {scroll_count} 次")
    
    def extract_posts_with_retry(self) -> List[Dict]:
        """带重试机制的帖子提取"""
        posts = []
        
        for attempt in range(self.config['max_retries']):
            try:
                logger.info(f"开始提取帖子 (尝试 {attempt + 1}/{self.config['max_retries']})...")
                
                # 等待内容加载
                if not self.wait_for_content_load():
                    if attempt < self.config['max_retries'] - 1:
                        logger.warning(f"内容加载失败，{self.config['retry_delay']}秒后重试...")
                        time.sleep(self.config['retry_delay'])
                        continue
                    else:
                        logger.error("多次尝试后仍无法加载内容")
                        break
                
                # 滚动加载更多内容
                self.scroll_and_load_comprehensive()
                
                # 查找帖子元素
                post_elements = self.find_post_elements_comprehensive()
                
                if not post_elements:
                    if attempt < self.config['max_retries'] - 1:
                        logger.warning(f"未找到帖子元素，{self.config['retry_delay']}秒后重试...")
                        time.sleep(self.config['retry_delay'])
                        continue
                    else:
                        logger.error("多次尝试后仍未找到帖子元素")
                        break
                
                logger.info(f"找到 {len(post_elements)} 个潜在帖子元素")
                
                # 提取帖子信息
                for i, post_element in enumerate(post_elements):
                    try:
                        post_data = self.extract_post_comprehensive(post_element)
                        if post_data:
                            posts.append(post_data)
                            if self.config['verbose_logging'] and len(posts) <= 5:
                                logger.info(f"✓ 提取帖子 {len(posts)}: {post_data['username']} - {post_data['content'][:50]}...")
                        
                        # 每处理100个元素输出进度
                        if (i + 1) % 100 == 0:
                            logger.info(f"已处理 {i + 1} 个元素，提取到 {len(posts)} 个有效帖子")
                            
                    except Exception as e:
                        logger.debug(f"提取第 {i+1} 个元素失败: {e}")
                        continue
                
                # 成功提取到内容，跳出重试循环
                if posts:
                    break
                    
            except Exception as e:
                logger.error(f"提取过程失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    logger.info(f"{self.config['retry_delay']}秒后重试...")
                    time.sleep(self.config['retry_delay'])
        
        logger.info(f"提取完成，总共获得 {len(posts)} 个中文帖子")
        return posts
    
    def save_debug_info(self) -> None:
        """保存调试信息"""
        if not (self.config['save_screenshots'] or self.config['save_page_source']):
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.config['save_screenshots']:
                screenshot_path = f"debug_final_scraper_{timestamp}.png"
                self.driver.save_screenshot(screenshot_path)
                logger.info(f"✓ 截图已保存: {screenshot_path}")
            
            if self.config['save_page_source']:
                source_path = f"debug_final_page_source_{timestamp}.html"
                with open(source_path, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                logger.info(f"✓ 页面源码已保存: {source_path}")
                
        except Exception as e:
            logger.warning(f"保存调试信息失败: {e}")
    
    def save_to_csv(self, filename: str = None) -> None:
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_final_{timestamp}.csv"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if not self.posts_data:
                    logger.warning("没有数据可保存")
                    return
                
                fieldnames = ['username', 'content', 'time', 'parsed_time', 'url', 'content_length', 'chinese_ratio']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for post in self.posts_data:
                    writer.writerow(post)
            
            logger.info(f"✓ 数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
    
    def save_to_json(self, filename: str = None) -> None:
        """保存数据到JSON文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_final_{timestamp}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # 添加元数据
            output_data = {
                "metadata": {
                    "scrape_time": datetime.now().isoformat(),
                    "target_url": self.target_url,
                    "total_posts": len(self.posts_data),
                    "config": self.config,
                    "time_range": {
                        "from": self.cutoff_date.isoformat(),
                        "to": datetime.now().isoformat()
                    }
                },
                "posts": self.posts_data
            }
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(output_data, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ 数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def run(self) -> None:
        """运行爬虫主程序"""
        try:
            logger.info("=" * 80)
            logger.info("长毛象爬虫最终版本启动")
            logger.info("=" * 80)
            logger.info(f"目标URL: {self.target_url}")
            logger.info(f"时间范围: {self.cutoff_date.strftime('%Y-%m-%d')} 至今")
            logger.info(f"中文检测阈值: 比例>{self.config['chinese_ratio_threshold']} 或 字符数>{self.config['chinese_char_threshold']}")
            logger.info(f"无头模式: {self.config['headless_mode']}")
            logger.info(f"反检测模式: {self.config['use_anti_detection']}")
            
            # 设置WebDriver
            self.setup_driver()
            
            # 访问目标页面
            logger.info("正在访问目标页面...")
            self.driver.get(self.target_url)
            
            # 检查登录状态
            if not self.check_login_status():
                logger.warning("页面可能需要登录，继续尝试...")
            
            # 提取帖子（带重试机制）
            self.posts_data = self.extract_posts_with_retry()
            
            # 保存调试信息
            self.save_debug_info()
            
            # 输出结果统计
            logger.info("=" * 80)
            logger.info("爬取结果统计")
            logger.info("=" * 80)
            logger.info(f"总共找到: {len(self.posts_data)} 个符合条件的中文帖子")
            
            if self.posts_data:
                # 统计信息
                avg_length = sum(post['content_length'] for post in self.posts_data) / len(self.posts_data)
                avg_chinese_ratio = sum(post['chinese_ratio'] for post in self.posts_data) / len(self.posts_data)
                
                logger.info(f"平均内容长度: {avg_length:.1f} 字符")
                logger.info(f"平均中文比例: {avg_chinese_ratio:.1%}")
                
                # 保存数据
                self.save_to_csv()
                self.save_to_json()
                
                # 显示示例数据
                logger.info("\n示例帖子:")
                for i, post in enumerate(self.posts_data[:3], 1):
                    logger.info(f"  {i}. {post['username']}: {post['content'][:80]}...")
                    logger.info(f"     时间: {post['time']}, 长度: {post['content_length']}, 中文比例: {post['chinese_ratio']:.1%}")
                
                logger.info(f"\n✅ 爬取成功！数据已保存到 data/raw/ 目录")
                
            else:
                logger.warning("❌ 没有找到符合条件的帖子")
                logger.info("\n可能的原因和建议:")
                logger.info("1. 调整中文检测阈值（降低 chinese_ratio_threshold）")
                logger.info("2. 扩大时间范围（增加 filter_days）")
                logger.info("3. 检查页面是否需要登录")
                logger.info("4. 运行诊断工具: python comprehensive_diagnostic.py")
                logger.info("5. 查看调试文件了解详细情况")
            
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
            self.save_debug_info()  # 即使失败也保存调试信息
            raise
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver 已关闭")


def main():
    """主函数"""
    print("长毛象爬虫最终版本")
    print("=" * 50)
    
    # 创建默认配置文件（如果不存在）
    config_file = 'scraper_config.json'
    if not os.path.exists(config_file):
        default_config = {
            "filter_days": 60,
            "chinese_ratio_threshold": 0.15,
            "chinese_char_threshold": 2,
            "headless_mode": True,
            "use_anti_detection": True,
            "max_scroll_attempts": 20,
            "verbose_logging": True,
            "save_screenshots": True,
            "save_page_source": True
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"✓ 已创建默认配置文件: {config_file}")
        except Exception as e:
            print(f"创建配置文件失败: {e}")
    
    # 询问运行模式
    mode = input("选择运行模式:\n1. 标准模式 (推荐)\n2. 可视化模式 (非无头)\n3. 调试模式 (详细日志)\n请输入 (1-3): ").strip()
    
    # 根据模式调整配置
    if mode == '2':
        # 可视化模式
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config['headless_mode'] = False
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("✓ 已切换到可视化模式")
    elif mode == '3':
        # 调试模式
        logging.getLogger().setLevel(logging.DEBUG)
        print("✓ 已启用调试模式")
    
    # 运行爬虫
    scraper = MastodonScraperFinal()
    
    try:
        scraper.run()
        return 0
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"\n爬虫运行失败: {e}")
        return 1


if __name__ == "__main__":
    import os
    exit(main())