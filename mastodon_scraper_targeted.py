#!/usr/bin/env python3
"""
长毛象爬虫 - 针对性解决方案
基于测试反馈的特定优化版本
"""

import re
import json
import csv
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
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
        logging.FileHandler('mastodon_scraper_targeted.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonScraperTargeted:
    """长毛象爬虫 - 针对性解决方案"""
    
    def __init__(self, base_url: str = "https://m.cmx.im"):
        self.base_url = base_url
        self.target_url = f"{base_url}/public/local"
        self.driver = None
        self.posts_data = []
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 宽松的时间过滤 - 近3个月
        self.cutoff_date = datetime.now() - timedelta(days=90)
        
        # 宽松的中文检测设置
        self.chinese_ratio_threshold = 0.05  # 5%的中文字符就算
        self.chinese_char_threshold = 1      # 1个中文字符就算
        self.min_content_length = 2          # 最小2个字符
        
    def setup_driver(self, headless: bool = True) -> None:
        """设置Chrome WebDriver"""
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument('--headless')
            
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # 反检测设置
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # 反检测脚本
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("Chrome WebDriver 设置完成")
            
        except Exception as e:
            logger.error(f"设置WebDriver失败: {e}")
            raise
    
    def is_chinese_content(self, text: str) -> bool:
        """宽松的中文内容检测"""
        if not text or len(text.strip()) < self.min_content_length:
            return False
        
        chinese_chars = self.chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        ratio = chinese_char_count / total_chars
        
        # 非常宽松的条件：只要有中文字符或比例超过5%
        return chinese_char_count >= self.chinese_char_threshold or ratio > self.chinese_ratio_threshold
    
    def parse_time_flexible(self, time_text: str) -> Optional[datetime]:
        """灵活的时间解析"""
        try:
            if not time_text:
                return None
                
            time_text = time_text.strip()
            current_time = datetime.now()
            
            # 中文相对时间
            if '分钟前' in time_text:
                match = re.search(r'(\d+)\s*分钟前', time_text)
                if match:
                    minutes = int(match.group(1))
                    return current_time - timedelta(minutes=minutes)
            
            if '小时前' in time_text:
                match = re.search(r'(\d+)\s*小时前', time_text)
                if match:
                    hours = int(match.group(1))
                    return current_time - timedelta(hours=hours)
            
            if '天前' in time_text:
                match = re.search(r'(\d+)\s*天前', time_text)
                if match:
                    days = int(match.group(1))
                    return current_time - timedelta(days=days)
            
            if '周前' in time_text:
                match = re.search(r'(\d+)\s*周前', time_text)
                if match:
                    weeks = int(match.group(1))
                    return current_time - timedelta(weeks=weeks)
            
            if '月前' in time_text:
                match = re.search(r'(\d+)\s*月前', time_text)
                if match:
                    months = int(match.group(1))
                    return current_time - timedelta(days=months*30)
            
            # 英文相对时间
            if 'm' in time_text and time_text.replace('m', '').strip().isdigit():
                minutes = int(time_text.replace('m', '').strip())
                return current_time - timedelta(minutes=minutes)
            
            if 'h' in time_text and time_text.replace('h', '').strip().isdigit():
                hours = int(time_text.replace('h', '').strip())
                return current_time - timedelta(hours=hours)
            
            if 'd' in time_text and time_text.replace('d', '').strip().isdigit():
                days = int(time_text.replace('d', '').strip())
                return current_time - timedelta(days=days)
            
            # 如果无法解析，返回当前时间（这样不会被时间过滤掉）
            return current_time
                
        except Exception as e:
            logger.debug(f"解析时间失败: {time_text}, 错误: {e}")
            # 解析失败时返回当前时间，避免被时间过滤
            return datetime.now()
    
    def wait_and_scroll_load(self) -> bool:
        """等待并滚动加载内容"""
        logger.info("等待页面加载并滚动获取内容...")
        
        try:
            # 等待应用容器
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "mastodon"))
            )
            logger.info("✓ Mastodon应用容器已加载")
            
            # 初始等待
            time.sleep(10)
            
            # 多次滚动加载内容
            for i in range(10):  # 增加滚动次数
                logger.info(f"第 {i+1} 次滚动...")
                
                # 滚动到底部
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                # 检查是否有文本元素
                all_elements = self.driver.find_elements(By.TAG_NAME, "div")
                text_elements = [el for el in all_elements if el.text.strip() and len(el.text.strip()) > 10]
                
                logger.info(f"   当前找到 {len(text_elements)} 个有文本的元素")
                
                # 如果找到足够的元素，继续滚动一会儿再停止
                if len(text_elements) > 50:
                    logger.info("找到足够的元素，再滚动几次...")
                    for j in range(5):
                        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(2)
                    break
            
            return True
            
        except TimeoutException:
            logger.error("页面加载超时")
            return False
        except Exception as e:
            logger.error(f"等待和滚动失败: {e}")
            return False
    
    def find_content_elements_flexible(self) -> List:
        """灵活查找内容元素"""
        logger.info("使用灵活策略查找内容元素...")
        
        # 基于测试结果，我们知道滚动后会有内容
        # 尝试查找所有可能包含帖子内容的元素
        
        potential_elements = []
        
        # 策略1: 查找所有有足够文本的div元素
        logger.info("策略1: 查找有文本的div元素...")
        all_divs = self.driver.find_elements(By.TAG_NAME, "div")
        
        for div in all_divs:
            try:
                text = div.text.strip()
                if text and len(text) >= 20:  # 至少20个字符
                    # 排除明显的导航、菜单等元素
                    if not any(keyword in text.lower() for keyword in ['navigation', 'menu', 'header', 'footer', 'sidebar']):
                        potential_elements.append(div)
            except:
                continue
        
        logger.info(f"策略1找到 {len(potential_elements)} 个潜在元素")
        
        # 策略2: 查找包含中文的元素
        logger.info("策略2: 查找包含中文的元素...")
        chinese_elements = []
        
        for element in potential_elements:
            try:
                text = element.text.strip()
                if self.is_chinese_content(text):
                    chinese_elements.append(element)
            except:
                continue
        
        logger.info(f"策略2找到 {len(chinese_elements)} 个中文元素")
        
        # 策略3: 查找可能的帖子容器
        logger.info("策略3: 查找帖子容器...")
        container_selectors = [
            "div[class*='item']",
            "div[class*='post']", 
            "div[class*='entry']",
            "div[class*='content']",
            "div[class*='message']",
            "div[class*='card']",
            "li",
            "section > div",
            "main > div > div"
        ]
        
        container_elements = []
        for selector in container_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    text = element.text.strip()
                    if text and len(text) >= 10 and self.is_chinese_content(text):
                        container_elements.append(element)
            except:
                continue
        
        logger.info(f"策略3找到 {len(container_elements)} 个容器元素")
        
        # 合并并去重
        all_candidates = list(set(chinese_elements + container_elements))
        
        # 按文本长度排序，优先处理内容较多的元素
        all_candidates.sort(key=lambda x: len(x.text.strip()) if x.text else 0, reverse=True)
        
        logger.info(f"总共找到 {len(all_candidates)} 个候选元素")
        
        return all_candidates
    
    def extract_post_from_element(self, element) -> Optional[Dict]:
        """从元素中提取帖子信息"""
        try:
            # 获取元素文本
            full_text = element.text.strip()
            if not full_text or len(full_text) < self.min_content_length:
                return None
            
            # 检查是否为中文内容
            if not self.is_chinese_content(full_text):
                return None
            
            # 尝试分离用户名和内容
            lines = full_text.split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            username = "未知用户"
            content = full_text
            time_text = "未知时间"
            post_url = self.target_url
            
            # 尝试识别用户名（通常在开头或包含@符号）
            for i, line in enumerate(lines):
                if line.startswith('@') or '用户' in line or len(line) < 50:
                    if i == 0 or (i == 1 and len(lines[0]) < 20):  # 用户名通常在前面
                        username = line
                        # 剩余部分作为内容
                        if i + 1 < len(lines):
                            content = '\n'.join(lines[i+1:])
                        break
            
            # 尝试识别时间信息
            time_patterns = [
                r'\d+\s*分钟前', r'\d+\s*小时前', r'\d+\s*天前', 
                r'\d+\s*周前', r'\d+\s*月前', r'\d+[mhd]'
            ]
            
            for line in lines:
                for pattern in time_patterns:
                    if re.search(pattern, line):
                        time_text = line
                        # 从内容中移除时间信息
                        content = content.replace(line, '').strip()
                        break
                if time_text != "未知时间":
                    break
            
            # 尝试从元素中找到链接
            try:
                links = element.find_elements(By.TAG_NAME, "a")
                for link in links:
                    href = link.get_attribute("href")
                    if href and ("/@" in href or "/status" in href or self.base_url in href):
                        post_url = href
                        break
            except:
                pass
            
            # 清理内容
            content = content.replace(username, '').strip()
            content = re.sub(r'\n+', '\n', content)  # 合并多个换行
            content = content[:1000]  # 限制长度
            
            if len(content) < self.min_content_length:
                return None
            
            # 解析时间
            post_time = self.parse_time_flexible(time_text)
            
            # 时间过滤（使用宽松的3个月范围）
            if post_time and post_time < self.cutoff_date:
                return None
            
            post_data = {
                'username': username,
                'content': content,
                'time': time_text,
                'parsed_time': post_time.isoformat() if post_time else None,
                'url': post_url,
                'content_length': len(content),
                'full_text_preview': full_text[:200] + "..." if len(full_text) > 200 else full_text
            }
            
            return post_data
            
        except Exception as e:
            logger.debug(f"提取帖子信息失败: {e}")
            return None
    
    def extract_posts_targeted(self) -> List[Dict]:
        """针对性的帖子提取"""
        logger.info("开始针对性帖子提取...")
        posts = []
        
        try:
            # 等待并滚动加载内容
            if not self.wait_and_scroll_load():
                logger.error("页面内容加载失败")
                return posts
            
            # 查找内容元素
            content_elements = self.find_content_elements_flexible()
            
            if not content_elements:
                logger.warning("未找到任何内容元素")
                return posts
            
            logger.info(f"开始处理 {len(content_elements)} 个内容元素...")
            
            # 处理每个元素
            processed_contents = set()  # 用于去重
            
            for i, element in enumerate(content_elements):
                try:
                    post_data = self.extract_post_from_element(element)
                    
                    if post_data:
                        # 简单去重：检查内容是否已经处理过
                        content_key = post_data['content'][:100]  # 使用前100个字符作为去重键
                        
                        if content_key not in processed_contents:
                            processed_contents.add(content_key)
                            posts.append(post_data)
                            
                            logger.info(f"✓ 提取帖子 {len(posts)}: {post_data['username']} - {post_data['content'][:50]}...")
                    
                    # 每处理50个元素输出进度
                    if (i + 1) % 50 == 0:
                        logger.info(f"已处理 {i + 1}/{len(content_elements)} 个元素，提取到 {len(posts)} 个帖子")
                        
                except Exception as e:
                    logger.debug(f"处理第 {i+1} 个元素失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"提取过程失败: {e}")
        
        logger.info(f"提取完成，总共获得 {len(posts)} 个中文帖子")
        return posts
    
    def save_to_csv(self, filename: str = None) -> None:
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_posts_targeted_{timestamp}.csv"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if not self.posts_data:
                    logger.warning("没有数据可保存")
                    return
                
                fieldnames = ['username', 'content', 'time', 'parsed_time', 'url', 'content_length', 'full_text_preview']
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
            filename = f"data/raw/mastodon_posts_targeted_{timestamp}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            output_data = {
                "metadata": {
                    "scrape_time": datetime.now().isoformat(),
                    "target_url": self.target_url,
                    "total_posts": len(self.posts_data),
                    "chinese_ratio_threshold": self.chinese_ratio_threshold,
                    "chinese_char_threshold": self.chinese_char_threshold,
                    "filter_days": 90,
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
    
    def save_debug_info(self) -> None:
        """保存调试信息"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存截图
            screenshot_path = f"debug_targeted_{timestamp}.png"
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"✓ 截图已保存: {screenshot_path}")
            
            # 保存页面源码
            source_path = f"debug_targeted_source_{timestamp}.html"
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            logger.info(f"✓ 页面源码已保存: {source_path}")
            
        except Exception as e:
            logger.warning(f"保存调试信息失败: {e}")
    
    def run(self, headless: bool = True) -> None:
        """运行爬虫"""
        try:
            logger.info("=" * 80)
            logger.info("长毛象爬虫 - 针对性解决方案")
            logger.info("=" * 80)
            logger.info(f"目标URL: {self.target_url}")
            logger.info(f"时间范围: {self.cutoff_date.strftime('%Y-%m-%d')} 至今 (90天)")
            logger.info(f"中文检测: 比例>{self.chinese_ratio_threshold} 或 字符数>{self.chinese_char_threshold}")
            logger.info(f"无头模式: {headless}")
            
            # 设置WebDriver
            self.setup_driver(headless=headless)
            
            # 访问目标页面
            logger.info("正在访问目标页面...")
            self.driver.get(self.target_url)
            
            # 提取帖子
            self.posts_data = self.extract_posts_targeted()
            
            # 保存调试信息
            self.save_debug_info()
            
            # 输出结果
            logger.info("=" * 80)
            logger.info("爬取结果")
            logger.info("=" * 80)
            
            if self.posts_data:
                logger.info(f"✅ 成功找到 {len(self.posts_data)} 个中文帖子！")
                
                # 统计信息
                avg_length = sum(post['content_length'] for post in self.posts_data) / len(self.posts_data)
                logger.info(f"平均内容长度: {avg_length:.1f} 字符")
                
                # 保存数据
                self.save_to_csv()
                self.save_to_json()
                
                # 显示示例
                logger.info("\n📝 示例帖子:")
                for i, post in enumerate(self.posts_data[:5], 1):
                    logger.info(f"  {i}. 用户: {post['username']}")
                    logger.info(f"     内容: {post['content'][:100]}...")
                    logger.info(f"     时间: {post['time']}")
                    logger.info(f"     长度: {post['content_length']} 字符")
                    logger.info("")
                
                logger.info(f"🎉 数据已保存到 data/raw/ 目录")
                
            else:
                logger.warning("❌ 没有找到符合条件的中文帖子")
                logger.info("请检查调试文件了解详情")
            
        except Exception as e:
            logger.error(f"爬虫运行失败: {e}")
            self.save_debug_info()
            raise
        finally:
            if self.driver:
                self.driver.quit()
                logger.info("WebDriver 已关闭")


def main():
    """主函数"""
    print("长毛象爬虫 - 针对性解决方案")
    print("基于测试反馈的优化版本")
    print("=" * 50)
    
    # 询问运行模式
    mode = input("选择运行模式:\n1. 无头模式 (推荐)\n2. 可视化模式\n请输入 (1-2): ").strip()
    
    headless_mode = mode != '2'
    
    scraper = MastodonScraperTargeted()
    
    try:
        scraper.run(headless=headless_mode)
        return 0
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"\n爬虫运行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())