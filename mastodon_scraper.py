#!/usr/bin/env python3
"""
长毛象中文站爬虫 - 爬取近两个月的中文帖子
目标网站: https://m.cmx.im/public/local
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
        logging.FileHandler('mastodon_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonScraper:
    """长毛象爬虫类"""
    
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
        """设置Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # 无头模式
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # 自动下载和设置ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Chrome WebDriver 设置完成")
            
        except Exception as e:
            logger.error(f"设置WebDriver失败: {e}")
            raise
    
    def is_chinese_content(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        if not text:
            return False
        chinese_chars = self.chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        # 如果中文字符占比超过20%或者中文字符数量超过3个，认为是中文内容
        return total_chars > 0 and ((chinese_char_count / total_chars) > 0.2 or chinese_char_count > 3)
    
    def parse_time(self, time_text: str) -> Optional[datetime]:
        """解析时间文本为datetime对象"""
        try:
            # 处理相对时间
            if '分钟前' in time_text:
                minutes = int(re.search(r'(\d+)', time_text).group(1))
                return datetime.now() - timedelta(minutes=minutes)
            elif '小时前' in time_text:
                hours = int(re.search(r'(\d+)', time_text).group(1))
                return datetime.now() - timedelta(hours=hours)
            elif '天前' in time_text:
                days = int(re.search(r'(\d+)', time_text).group(1))
                return datetime.now() - timedelta(days=days)
            elif '周前' in time_text:
                weeks = int(re.search(r'(\d+)', time_text).group(1))
                return datetime.now() - timedelta(weeks=weeks)
            elif '月前' in time_text:
                months = int(re.search(r'(\d+)', time_text).group(1))
                return datetime.now() - timedelta(days=months*30)
            else:
                # 尝试解析绝对时间格式
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m-%d %H:%M']:
                    try:
                        return datetime.strptime(time_text, fmt)
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.warning(f"解析时间失败: {time_text}, 错误: {e}")
        
        return None
    
    def scroll_and_load_posts(self, max_scrolls: int = 50) -> None:
        """滚动页面加载更多帖子"""
        logger.info("开始滚动加载帖子...")
        
        scroll_count = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        while scroll_count < max_scrolls:
            # 滚动到页面底部
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # 等待新内容加载
            time.sleep(2)
            
            # 检查是否有新内容加载
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                logger.info("没有更多内容可加载")
                break
                
            last_height = new_height
            scroll_count += 1
            
            if scroll_count % 10 == 0:
                logger.info(f"已滚动 {scroll_count} 次")
    
    def extract_posts(self) -> List[Dict]:
        """提取帖子信息"""
        logger.info("开始提取帖子信息...")
        posts = []
        
        try:
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article, .status, [data-testid='status']"))
            )
            
            # 滚动加载更多内容
            self.scroll_and_load_posts()
            
            # 查找帖子元素 - 尝试多种选择器
            post_selectors = [
                "article",
                ".status",
                "[data-testid='status']",
                ".status__wrapper",
                ".detailed-status"
            ]
            
            post_elements = []
            for selector in post_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    post_elements = elements
                    logger.info(f"找到 {len(elements)} 个帖子元素 (使用选择器: {selector})")
                    break
            
            if not post_elements:
                logger.warning("未找到帖子元素")
                return posts
            
            for i, post_element in enumerate(post_elements):
                try:
                    post_data = self.extract_single_post(post_element)
                    if post_data:
                        posts.append(post_data)
                        
                        # 每处理100个帖子输出一次进度
                        if (i + 1) % 100 == 0:
                            logger.info(f"已处理 {i + 1} 个帖子")
                            
                except Exception as e:
                    logger.warning(f"提取第 {i+1} 个帖子失败: {e}")
                    continue
                    
        except TimeoutException:
            logger.error("页面加载超时")
        except Exception as e:
            logger.error(f"提取帖子时发生错误: {e}")
        
        logger.info(f"总共提取到 {len(posts)} 个帖子")
        return posts
    
    def extract_single_post(self, post_element) -> Optional[Dict]:
        """提取单个帖子的信息"""
        try:
            post_data = {}
            
            # 提取用户名
            username_selectors = [
                ".display-name__account",
                ".status__display-name",
                "[data-testid='username']",
                ".account__display-name",
                "a[href*='/@']"
            ]
            
            username = None
            for selector in username_selectors:
                try:
                    username_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    username = username_element.text.strip()
                    if username:
                        break
                except NoSuchElementException:
                    continue
            
            # 提取帖子内容
            content_selectors = [
                ".status__content",
                ".status__content__text",
                "[data-testid='statusContent']",
                ".detailed-status__content",
                ".status-content"
            ]
            
            content = None
            for selector in content_selectors:
                try:
                    content_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    content = content_element.text.strip()
                    if content:
                        break
                except NoSuchElementException:
                    continue
            
            # 提取时间
            time_selectors = [
                "time",
                ".status__relative-time",
                "[data-testid='timestamp']",
                ".detailed-status__datetime"
            ]
            
            time_text = None
            post_url = None
            for selector in time_selectors:
                try:
                    time_element = post_element.find_element(By.CSS_SELECTOR, selector)
                    time_text = time_element.text.strip()
                    
                    # 尝试获取帖子链接
                    if time_element.tag_name == 'a' or time_element.find_elements(By.TAG_NAME, 'a'):
                        link_element = time_element if time_element.tag_name == 'a' else time_element.find_element(By.TAG_NAME, 'a')
                        post_url = link_element.get_attribute('href')
                    
                    if time_text:
                        break
                except NoSuchElementException:
                    continue
            
            # 检查是否有有效内容
            if not content or not username:
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
                'time': time_text,
                'parsed_time': post_time.isoformat() if post_time else None,
                'url': post_url or self.target_url
            }
            
            return post_data
            
        except Exception as e:
            logger.warning(f"提取单个帖子信息失败: {e}")
            return None
    
    def save_to_csv(self, filename: str = None) -> None:
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_{timestamp}.csv"
        
        try:
            # 确保目录存在
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
            filename = f"data/raw/mastodon_posts_{timestamp}.json"
        
        try:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.posts_data, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def run(self) -> None:
        """运行爬虫"""
        try:
            logger.info("开始爬取长毛象中文站...")
            logger.info(f"目标URL: {self.target_url}")
            logger.info(f"时间范围: {self.cutoff_date.strftime('%Y-%m-%d')} 至今")
            
            # 设置WebDriver
            self.setup_driver()
            
            # 访问目标页面
            logger.info("正在访问目标页面...")
            self.driver.get(self.target_url)
            
            # 等待页面加载
            time.sleep(5)
            
            # 提取帖子
            self.posts_data = self.extract_posts()
            
            # 过滤中文帖子
            chinese_posts = [post for post in self.posts_data if self.is_chinese_content(post.get('content', ''))]
            self.posts_data = chinese_posts
            
            logger.info(f"找到 {len(self.posts_data)} 个中文帖子")
            
            # 保存数据
            if self.posts_data:
                self.save_to_csv()
                self.save_to_json()
            else:
                logger.warning("没有找到符合条件的帖子")
            
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver 已关闭")


def main():
    """主函数"""
    scraper = MastodonScraper()
    scraper.run()


if __name__ == "__main__":
    main()