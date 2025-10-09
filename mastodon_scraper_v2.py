#!/usr/bin/env python3
"""
长毛象中文站爬虫 v2.0 - 改进版本
目标网站: https://m.cmx.im/public/local
修复了页面加载和元素定位问题
"""

import re
import json
import csv
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
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
        logging.FileHandler('mastodon_scraper_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonScraperV2:
    """长毛象爬虫类 v2.0 - 改进版本"""
    
    def __init__(self, base_url: str = "https://m.cmx.im"):
        self.base_url = base_url
        self.target_url = f"{base_url}/public/local"
        self.driver = None
        self.posts_data = []
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 时间过滤 - 近两个月
        self.cutoff_date = datetime.now() - timedelta(days=60)
        
    def setup_driver(self) -> None:
        """设置Chrome WebDriver - 改进版本"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # 禁用图片和CSS加载以提高速度
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # 设置反检测
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome WebDriver 设置完成")
            
        except Exception as e:
            logger.error(f"设置WebDriver失败: {e}")
            raise
    
    def is_chinese_content(self, text: str) -> bool:
        """检查文本是否包含中文字符 - 改进版本"""
        if not text:
            return False
        chinese_chars = self.chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        # 如果中文字符占比超过15%或者中文字符数量超过2个，认为是中文内容
        return total_chars > 0 and ((chinese_char_count / total_chars) > 0.15 or chinese_char_count > 2)
    
    def parse_time(self, time_text: str) -> Optional[datetime]:
        """解析时间文本为datetime对象 - 改进版本"""
        try:
            if not time_text:
                return None
                
            time_text = time_text.strip()
            
            # 处理相对时间
            if '分钟前' in time_text or 'minutes ago' in time_text or 'm' == time_text[-1:]:
                minutes_match = re.search(r'(\d+)', time_text)
                if minutes_match:
                    minutes = int(minutes_match.group(1))
                    return datetime.now() - timedelta(minutes=minutes)
                    
            elif '小时前' in time_text or 'hours ago' in time_text or 'h' == time_text[-1:]:
                hours_match = re.search(r'(\d+)', time_text)
                if hours_match:
                    hours = int(hours_match.group(1))
                    return datetime.now() - timedelta(hours=hours)
                    
            elif '天前' in time_text or 'days ago' in time_text or 'd' == time_text[-1:]:
                days_match = re.search(r'(\d+)', time_text)
                if days_match:
                    days = int(days_match.group(1))
                    return datetime.now() - timedelta(days=days)
                    
            elif '周前' in time_text or 'weeks ago' in time_text or 'w' == time_text[-1:]:
                weeks_match = re.search(r'(\d+)', time_text)
                if weeks_match:
                    weeks = int(weeks_match.group(1))
                    return datetime.now() - timedelta(weeks=weeks)
                    
            elif '月前' in time_text or 'months ago' in time_text:
                months_match = re.search(r'(\d+)', time_text)
                if months_match:
                    months = int(months_match.group(1))
                    return datetime.now() - timedelta(days=months*30)
            else:
                # 尝试解析绝对时间格式
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m-%d %H:%M', '%H:%M']:
                    try:
                        return datetime.strptime(time_text, fmt)
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.debug(f"解析时间失败: {time_text}, 错误: {e}")
        
        return None
    
    def wait_for_content_load(self, timeout: int = 30) -> bool:
        """等待页面内容加载完成"""
        logger.info("等待页面内容加载...")
        
        try:
            # 等待应用容器加载
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.ID, "mastodon"))
            )
            logger.info("✓ Mastodon应用容器已加载")
            
            # 等待更长时间让JavaScript渲染内容
            time.sleep(10)
            
            # 尝试等待任何可能的内容元素
            content_selectors = [
                "article", ".status", "[role='article']", 
                "div[class*='status']", "[data-testid]",
                ".timeline-item", "div[class*='post']"
            ]
            
            for selector in content_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.info(f"✓ 找到内容元素: {selector} ({len(elements)}个)")
                        return True
                except:
                    continue
            
            # 如果没有找到标准元素，尝试滚动触发加载
            logger.info("尝试滚动触发内容加载...")
            for i in range(5):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # 再次检查是否有内容
                for selector in content_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            logger.info(f"✓ 滚动后找到内容: {selector} ({len(elements)}个)")
                            return True
                    except:
                        continue
            
            return False
            
        except TimeoutException:
            logger.error("页面加载超时")
            return False
        except Exception as e:
            logger.error(f"等待内容加载失败: {e}")
            return False
    
    def find_post_elements(self) -> List:
        """查找帖子元素 - 改进版本"""
        logger.info("查找帖子元素...")
        
        # 扩展的选择器列表，按优先级排序
        selectors = [
            # Mastodon标准选择器
            "article",
            ".status",
            ".status__wrapper",
            "[data-testid='status']",
            ".detailed-status",
            
            # 通用选择器
            "div[role='article']",
            "[role='listitem']",
            ".timeline-item",
            
            # 类名包含关键词的选择器
            "div[class*='status']",
            "div[class*='post']",
            "div[class*='toot']",
            "div[class*='timeline']",
            
            # 更宽泛的选择器
            "div[data-testid]",
            "div[class*='item']",
            
            # 最后尝试的选择器
            "main div > div",  # 主要内容区域的直接子元素
            ".app-body div[class]",  # 应用主体中有类名的div
        ]
        
        for selector in selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    # 过滤掉明显不是帖子的元素
                    filtered_elements = []
                    for element in elements:
                        try:
                            text = element.text.strip()
                            # 基本过滤：有文本内容且不是太短
                            if text and len(text) > 10:
                                filtered_elements.append(element)
                        except:
                            continue
                    
                    if filtered_elements:
                        logger.info(f"✓ 使用选择器 '{selector}' 找到 {len(filtered_elements)} 个有效帖子元素")
                        return filtered_elements
                else:
                    logger.debug(f"选择器 '{selector}' 未找到元素")
            except Exception as e:
                logger.debug(f"选择器 '{selector}' 查找失败: {e}")
        
        logger.warning("未找到任何帖子元素")
        return []
    
    def extract_single_post(self, post_element) -> Optional[Dict]:
        """提取单个帖子的信息 - 改进版本"""
        try:
            post_data = {}
            
            # 获取元素的所有文本内容
            full_text = post_element.text.strip()
            if not full_text or len(full_text) < 10:
                return None
            
            # 尝试提取用户名 - 扩展选择器
            username_selectors = [
                ".display-name__account", ".status__display-name", "[data-testid='username']",
                ".account__display-name", "a[href*='/@']", ".username", ".author",
                "span[class*='username']", "div[class*='username']", "a[class*='username']",
                "span[class*='display-name']", "div[class*='display-name']"
            ]
            
            username = None
            for selector in username_selectors:
                try:
                    username_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    username = username_element.text.strip()
                    if username and len(username) > 0:
                        break
                except NoSuchElementException:
                    continue
            
            # 如果没找到用户名，尝试从链接中提取
            if not username:
                try:
                    links = post_element.find_elements(By.TAG_NAME, "a")
                    for link in links:
                        href = link.get_attribute("href")
                        if href and "/@" in href:
                            username = href.split("/@")[-1].split("/")[0]
                            if username:
                                username = f"@{username}"
                                break
                except:
                    pass
            
            # 如果还是没找到用户名，使用默认值
            if not username:
                username = "未知用户"
            
            # 提取帖子内容 - 扩展选择器
            content_selectors = [
                ".status__content", ".status__content__text", "[data-testid='statusContent']",
                ".detailed-status__content", ".status-content", ".content", ".post-content",
                "div[class*='content']", "p", ".text"
            ]
            
            content = None
            for selector in content_selectors:
                try:
                    content_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    content = content_element.text.strip()
                    if content and len(content) > 0:
                        break
                except NoSuchElementException:
                    continue
            
            # 如果没找到特定内容元素，使用整个元素的文本
            if not content:
                content = full_text
            
            # 清理内容，移除用户名等无关信息
            if content and username != "未知用户":
                clean_username = username.replace("@", "")
                content = content.replace(username, "").replace(clean_username, "").strip()
            
            # 提取时间 - 扩展选择器
            time_selectors = [
                "time", ".status__relative-time", "[data-testid='timestamp']",
                ".detailed-status__datetime", ".timestamp", ".time", ".date",
                "span[class*='time']", "div[class*='time']", "a[class*='time']"
            ]
            
            time_text = None
            post_url = None
            for selector in time_selectors:
                try:
                    time_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    time_text = time_element.text.strip()
                    
                    # 尝试获取帖子链接
                    if time_element.tag_name == 'a':
                        post_url = time_element.get_attribute('href')
                    else:
                        parent_link = time_element.find_element(By.XPATH, ".//ancestor::a[1]")
                        if parent_link:
                            post_url = parent_link.get_attribute('href')
                    
                    if time_text:
                        break
                except NoSuchElementException:
                    continue
                except:
                    pass
            
            # 检查是否有有效内容
            if not content or len(content.strip()) < 5:
                return None
            
            # 检查是否为中文内容
            if not self.is_chinese_content(content):
                return None
            
            # 解析时间并检查是否在时间范围内
            post_time = self.parse_time(time_text) if time_text else None
            if post_time and post_time < self.cutoff_date:
                return None
            
            # 构建完整的帖子URL
            if post_url and not post_url.startswith('http'):
                post_url = urljoin(self.base_url, post_url)
            
            post_data = {
                'username': username,
                'content': content,
                'time': time_text or "未知时间",
                'parsed_time': post_time.isoformat() if post_time else None,
                'url': post_url or self.target_url
            }
            
            return post_data
            
        except Exception as e:
            logger.debug(f"提取单个帖子信息失败: {e}")
            return None
    
    def scroll_and_load_posts(self, max_scrolls: int = 20) -> None:
        """滚动页面加载更多帖子 - 改进版本"""
        logger.info("开始滚动加载帖子...")
        
        scroll_count = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        no_change_count = 0
        
        while scroll_count < max_scrolls and no_change_count < 3:
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # 等待新内容加载
            time.sleep(3)
            
            # 检查是否有新内容加载
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                no_change_count += 1
                logger.info(f"页面高度未变化 ({no_change_count}/3)")
            else:
                no_change_count = 0
                logger.info(f"页面高度变化: {last_height} -> {new_height}")
            
            last_height = new_height
            scroll_count += 1
            
            if scroll_count % 5 == 0:
                logger.info(f"已滚动 {scroll_count} 次")
        
        if no_change_count >= 3:
            logger.info("连续3次无新内容，停止滚动")
        else:
            logger.info(f"达到最大滚动次数 {max_scrolls}")
    
    def extract_posts(self) -> List[Dict]:
        """提取帖子信息 - 改进版本"""
        logger.info("开始提取帖子信息...")
        posts = []
        
        try:
            # 等待页面内容加载
            if not self.wait_for_content_load():
                logger.error("页面内容加载失败")
                return posts
            
            # 滚动加载更多内容
            self.scroll_and_load_posts()
            
            # 查找帖子元素
            post_elements = self.find_post_elements()
            
            if not post_elements:
                logger.warning("未找到帖子元素")
                
                # 保存页面源码用于调试
                with open("debug_no_posts_found.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                logger.info("页面源码已保存到 debug_no_posts_found.html")
                
                return posts
            
            logger.info(f"找到 {len(post_elements)} 个潜在帖子元素")
            
            for i, post_element in enumerate(post_elements):
                try:
                    post_data = self.extract_single_post(post_element)
                    if post_data:
                        posts.append(post_data)
                        logger.info(f"成功提取第 {len(posts)} 个帖子: {post_data['username']} - {post_data['content'][:50]}...")
                        
                    # 每处理50个元素输出一次进度
                    if (i + 1) % 50 == 0:
                        logger.info(f"已处理 {i + 1} 个元素，提取到 {len(posts)} 个有效帖子")
                        
                except Exception as e:
                    logger.debug(f"提取第 {i+1} 个元素失败: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"提取帖子时发生错误: {e}")
        
        logger.info(f"总共提取到 {len(posts)} 个中文帖子")
        return posts
    
    def save_to_csv(self, filename: str = None) -> None:
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_v2_{timestamp}.csv"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if not self.posts_data:
                    logger.warning("没有数据可保存")
                    return
                
                fieldnames = ['username', 'content', 'time', 'parsed_time', 'url']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for post in self.posts_data:
                    writer.writerow(post)
            
            logger.info(f"数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
    
    def save_to_json(self, filename: str = None) -> None:
        """保存数据到JSON文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_v2_{timestamp}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.posts_data, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def run(self) -> None:
        """运行爬虫 - 改进版本"""
        try:
            logger.info("开始爬取长毛象中文站... (v2.0)")
            logger.info(f"目标URL: {self.target_url}")
            logger.info(f"时间范围: {self.cutoff_date.strftime('%Y-%m-%d')} 至今")
            
            # 设置WebDriver
            self.setup_driver()
            
            # 访问目标页面
            logger.info("正在访问目标页面...")
            self.driver.get(self.target_url)
            
            # 提取帖子
            self.posts_data = self.extract_posts()
            
            logger.info(f"找到 {len(self.posts_data)} 个符合条件的中文帖子")
            
            # 保存数据
            if self.posts_data:
                self.save_to_csv()
                self.save_to_json()
                
                # 显示一些示例数据
                logger.info("示例帖子:")
                for i, post in enumerate(self.posts_data[:3]):
                    logger.info(f"  {i+1}. {post['username']}: {post['content'][:100]}...")
            else:
                logger.warning("没有找到符合条件的帖子")
                logger.info("可能的原因:")
                logger.info("1. 页面结构发生变化")
                logger.info("2. 需要登录才能查看内容")
                logger.info("3. 网站有反爬虫机制")
                logger.info("4. 时间过滤太严格")
                logger.info("请检查 debug_no_posts_found.html 文件以获取更多信息")
            
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver 已关闭")


def main():
    """主函数"""
    scraper = MastodonScraperV2()
    scraper.run()


if __name__ == "__main__":
    main()