#!/usr/bin/env python3
"""
简化连接测试 - 不依赖浏览器的基本测试
"""

import re
import json
import requests
from datetime import datetime, timedelta

def test_basic_connection():
    """测试基本网络连接和API"""
    print("=" * 50)
    print("长毛象网站连接测试")
    print("=" * 50)
    
    base_url = "https://m.cmx.im"
    
    # 1. 测试主页连接
    print("\n1. 测试主页连接...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"   ✓ 主页响应状态: {response.status_code}")
        print(f"   ✓ 响应大小: {len(response.text)} 字符")
        
        # 检查是否包含Mastodon相关内容
        if "mastodon" in response.text.lower():
            print("   ✓ 页面包含Mastodon内容")
        else:
            print("   ⚠ 页面不包含Mastodon内容")
            
    except Exception as e:
        print(f"   ✗ 主页连接失败: {e}")
        return False
    
    # 2. 测试公共页面
    print("\n2. 测试公共时间线页面...")
    try:
        public_url = f"{base_url}/public/local"
        response = requests.get(public_url, timeout=10)
        print(f"   ✓ 公共页面状态: {response.status_code}")
        
        # 检查页面内容
        if "initial-state" in response.text:
            print("   ✓ 页面包含初始状态数据")
            
            # 尝试提取初始状态JSON
            try:
                import re
                pattern = r'<script id="initial-state" type="application/json">(.*?)</script>'
                match = re.search(pattern, response.text, re.DOTALL)
                if match:
                    initial_state = json.loads(match.group(1))
                    print(f"   ✓ 成功解析初始状态数据")
                    print(f"   ✓ 实例域名: {initial_state.get('meta', {}).get('domain', 'unknown')}")
                    print(f"   ✓ 实例版本: {initial_state.get('meta', {}).get('version', 'unknown')}")
                else:
                    print("   ⚠ 无法提取初始状态JSON")
            except Exception as e:
                print(f"   ⚠ 解析初始状态失败: {e}")
        else:
            print("   ⚠ 页面不包含初始状态数据")
            
    except Exception as e:
        print(f"   ✗ 公共页面连接失败: {e}")
    
    # 3. 测试实例信息API
    print("\n3. 测试实例信息API...")
    try:
        api_url = f"{base_url}/api/v1/instance"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(api_url, headers=headers, timeout=10)
        print(f"   ✓ API响应状态: {response.status_code}")
        
        if response.status_code == 200:
            instance_info = response.json()
            print(f"   ✓ 实例标题: {instance_info.get('title', 'unknown')}")
            print(f"   ✓ 实例描述: {instance_info.get('short_description', 'unknown')[:50]}...")
            print(f"   ✓ 用户数量: {instance_info.get('stats', {}).get('user_count', 'unknown')}")
            print(f"   ✓ 帖子数量: {instance_info.get('stats', {}).get('status_count', 'unknown')}")
        else:
            print(f"   ⚠ API响应异常: {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ 实例API测试失败: {e}")
    
    # 4. 测试时间线API（可能需要认证）
    print("\n4. 测试时间线API...")
    try:
        timeline_url = f"{base_url}/api/v1/timelines/public"
        params = {'local': 'true', 'limit': 5}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        response = requests.get(timeline_url, params=params, headers=headers, timeout=10)
        print(f"   ✓ 时间线API状态: {response.status_code}")
        
        if response.status_code == 200:
            timeline_data = response.json()
            print(f"   ✓ 获取到 {len(timeline_data)} 条时间线数据")
            
            # 检查是否有中文内容
            chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
            chinese_posts = 0
            
            for post in timeline_data:
                content = post.get('content', '')
                if content:
                    # 清理HTML标签
                    clean_content = re.sub(r'<[^>]+>', '', content)
                    if chinese_pattern.search(clean_content):
                        chinese_posts += 1
            
            print(f"   ✓ 其中包含中文的帖子: {chinese_posts} 条")
            
            if chinese_posts > 0:
                print("   🎉 发现中文内容！API方案可行")
            else:
                print("   ⚠ 未发现中文内容，可能需要更多数据")
                
        elif response.status_code == 401:
            print("   ⚠ 时间线API需要认证")
        elif response.status_code == 422:
            print("   ⚠ 时间线API请求格式错误或需要特殊权限")
        else:
            print(f"   ⚠ 时间线API响应异常: {response.status_code}")
            
    except Exception as e:
        print(f"   ✗ 时间线API测试失败: {e}")
    
    # 5. 中文检测功能测试
    print("\n5. 测试中文检测功能...")
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    test_cases = [
        ("这是一个中文测试", True),
        ("This is English text", False),
        ("中英混合 mixed text 测试", True),
        ("Hello 世界", True),
        ("今天天气很好！😊", True),
        ("", False),
        ("123456", False)
    ]
    
    for text, expected in test_cases:
        chinese_chars = chinese_pattern.findall(text)
        chinese_char_count = sum(len(chars) for chars in chinese_chars)
        total_chars = len(text.strip())
        
        is_chinese = total_chars > 0 and ((chinese_char_count / total_chars) > 0.15 or chinese_char_count > 2)
        result = "✓" if is_chinese == expected else "✗"
        print(f"   {result} '{text}' -> 中文: {is_chinese} (期望: {expected})")
    
    # 6. 总结和建议
    print("\n" + "=" * 50)
    print("测试总结和建议")
    print("=" * 50)
    
    print("\n基于测试结果，建议您:")
    print("1. 如果时间线API可用 -> 使用 mastodon_api_scraper.py")
    print("2. 如果API不可用但网页正常 -> 使用 mastodon_scraper_v2.py")
    print("3. 如果都有问题 -> 查看 问题解决方案.md")
    
    print("\n可用的脚本文件:")
    print("- mastodon_scraper_v2.py (改进版Selenium爬虫)")
    print("- mastodon_api_scraper.py (API方式爬虫)")
    print("- debug_scraper.py (调试版本)")
    print("- 问题解决方案.md (详细故障排除指南)")
    
    return True


if __name__ == "__main__":
    test_basic_connection()