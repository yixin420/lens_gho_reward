#!/usr/bin/env python3
"""
长毛象API爬虫 - 备用方案
尝试通过API接口获取数据，避免浏览器依赖
"""

import re
import json
import csv
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import urljoin

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mastodon_api_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MastodonAPIScraper:
    """长毛象API爬虫类"""
    
    def __init__(self, base_url: str = "https://m.cmx.im"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.posts_data = []
        
        # 中文字符正则表达式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        
        # 时间过滤 - 近两个月
        self.cutoff_date = datetime.now() - timedelta(days=60)
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def is_chinese_content(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        if not text:
            return False
        chinese_chars = self.chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        return total_chars > 0 and ((chinese_char_count / total_chars) > 0.15 or chinese_char_count > 2)
    
    def parse_mastodon_time(self, time_str: str) -> Optional[datetime]:
        """解析Mastodon API返回的时间格式"""
        try:
            # Mastodon API通常返回ISO 8601格式
            # 例如: "2024-10-09T10:30:00.000Z"
            if time_str.endswith('Z'):
                time_str = time_str[:-1] + '+00:00'
            
            # 尝试多种ISO格式
            formats = [
                '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
                    
        except Exception as e:
            logger.debug(f"解析时间失败: {time_str}, 错误: {e}")
        
        return None
    
    def test_api_endpoints(self) -> Dict[str, bool]:
        """测试各种API端点的可用性"""
        logger.info("测试API端点可用性...")
        
        endpoints = {
            'instance_info': '/api/v1/instance',
            'public_timeline': '/api/v1/timelines/public',
            'local_timeline': '/api/v1/timelines/public?local=true',
            'trends': '/api/v1/trends/statuses',
        }
        
        results = {}
        
        for name, endpoint in endpoints.items():
            try:
                url = f"{self.base_url}{endpoint}"
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    results[name] = True
                    logger.info(f"✓ {name}: 可用 (状态码: {response.status_code})")
                    
                    # 如果是时间线端点，检查返回数据
                    if 'timeline' in name:
                        try:
                            data = response.json()
                            if isinstance(data, list) and len(data) > 0:
                                logger.info(f"  返回 {len(data)} 条数据")
                            else:
                                logger.info(f"  返回数据为空或格式异常")
                        except:
                            logger.info(f"  无法解析JSON响应")
                            
                elif response.status_code == 401:
                    results[name] = False
                    logger.warning(f"✗ {name}: 需要认证 (状态码: {response.status_code})")
                else:
                    results[name] = False
                    logger.warning(f"✗ {name}: 不可用 (状态码: {response.status_code})")
                    
            except Exception as e:
                results[name] = False
                logger.error(f"✗ {name}: 请求失败 - {e}")
        
        return results
    
    def get_public_timeline(self, local_only: bool = True, limit: int = 40) -> List[Dict]:
        """获取公共时间线数据"""
        logger.info(f"获取公共时间线数据 (local_only={local_only}, limit={limit})...")
        
        params = {
            'limit': limit
        }
        
        if local_only:
            params['local'] = 'true'
        
        try:
            url = f"{self.api_base}/timelines/public"
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"成功获取 {len(data)} 条时间线数据")
                return data
            elif response.status_code == 401:
                logger.error("API需要认证，无法获取数据")
                return []
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}")
                logger.debug(f"响应内容: {response.text[:500]}")
                return []
                
        except Exception as e:
            logger.error(f"获取时间线数据失败: {e}")
            return []
    
    def parse_status(self, status: Dict) -> Optional[Dict]:
        """解析单条状态数据"""
        try:
            # 提取基本信息
            status_id = status.get('id', '')
            content = status.get('content', '')
            created_at = status.get('created_at', '')
            
            # 清理HTML标签
            if content:
                import html
                content = html.unescape(content)
                content = re.sub(r'<[^>]+>', '', content)
                content = content.strip()
            
            # 检查是否为中文内容
            if not content or not self.is_chinese_content(content):
                return None
            
            # 解析时间
            post_time = self.parse_mastodon_time(created_at)
            if post_time and post_time < self.cutoff_date:
                return None
            
            # 提取用户信息
            account = status.get('account', {})
            username = account.get('acct', '') or account.get('username', '')
            if username:
                username = f"@{username}"
            else:
                username = "未知用户"
            
            # 构建帖子URL
            post_url = status.get('url', '') or f"{self.base_url}/@{username}/{status_id}"
            
            return {
                'username': username,
                'content': content,
                'time': created_at,
                'parsed_time': post_time.isoformat() if post_time else None,
                'url': post_url
            }
            
        except Exception as e:
            logger.debug(f"解析状态数据失败: {e}")
            return None
    
    def scrape_with_pagination(self, max_requests: int = 10) -> List[Dict]:
        """使用分页获取更多数据"""
        logger.info(f"开始分页获取数据 (最多 {max_requests} 次请求)...")
        
        all_posts = []
        max_id = None
        
        for i in range(max_requests):
            logger.info(f"第 {i+1}/{max_requests} 次请求...")
            
            params = {
                'limit': 40,
                'local': 'true'
            }
            
            if max_id:
                params['max_id'] = max_id
            
            try:
                url = f"{self.api_base}/timelines/public"
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code != 200:
                    logger.warning(f"请求失败，状态码: {response.status_code}")
                    break
                
                data = response.json()
                if not data:
                    logger.info("没有更多数据")
                    break
                
                # 处理数据
                batch_posts = []
                for status in data:
                    post_data = self.parse_status(status)
                    if post_data:
                        batch_posts.append(post_data)
                
                all_posts.extend(batch_posts)
                logger.info(f"本批次获取到 {len(batch_posts)} 个中文帖子")
                
                # 获取下一页的max_id
                if data:
                    max_id = data[-1].get('id')
                else:
                    break
                
                # 避免请求过快
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"分页请求失败: {e}")
                break
        
        logger.info(f"分页获取完成，总共获得 {len(all_posts)} 个中文帖子")
        return all_posts
    
    def save_to_csv(self, filename: str = None) -> None:
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/mastodon_api_posts_{timestamp}.csv"
        
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
            filename = f"data/raw/mastodon_api_posts_{timestamp}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(self.posts_data, jsonfile, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
    
    def run(self) -> None:
        """运行API爬虫"""
        try:
            logger.info("开始使用API方式爬取长毛象中文站...")
            logger.info(f"目标URL: {self.base_url}")
            logger.info(f"时间范围: {self.cutoff_date.strftime('%Y-%m-%d')} 至今")
            
            # 测试API端点
            api_status = self.test_api_endpoints()
            
            if not any(api_status.values()):
                logger.error("所有API端点都不可用")
                return
            
            # 尝试获取数据
            if api_status.get('local_timeline', False):
                logger.info("使用本地时间线API获取数据...")
                self.posts_data = self.scrape_with_pagination()
            elif api_status.get('public_timeline', False):
                logger.info("使用公共时间线API获取数据...")
                timeline_data = self.get_public_timeline(local_only=False)
                self.posts_data = []
                for status in timeline_data:
                    post_data = self.parse_status(status)
                    if post_data:
                        self.posts_data.append(post_data)
            else:
                logger.error("没有可用的时间线API")
                return
            
            logger.info(f"找到 {len(self.posts_data)} 个符合条件的中文帖子")
            
            # 保存数据
            if self.posts_data:
                self.save_to_csv()
                self.save_to_json()
                
                # 显示示例数据
                logger.info("示例帖子:")
                for i, post in enumerate(self.posts_data[:3]):
                    logger.info(f"  {i+1}. {post['username']}: {post['content'][:100]}...")
            else:
                logger.warning("没有找到符合条件的帖子")
                logger.info("可能的原因:")
                logger.info("1. API需要认证")
                logger.info("2. 时间过滤太严格")
                logger.info("3. 中文内容检测过于严格")
                logger.info("4. 该实例没有公开的中文内容")
            
        except Exception as e:
            logger.error(f"API爬虫运行失败: {e}")
            raise


def main():
    """主函数"""
    scraper = MastodonAPIScraper()
    scraper.run()


if __name__ == "__main__":
    main()