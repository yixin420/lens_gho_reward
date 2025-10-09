#!/usr/bin/env python3
"""
长毛象爬虫调试版本 - 用于诊断页面加载和元素定位问题
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_page_structure():
    """调试页面结构，找出正确的选择器"""
    
    # 设置Chrome选项
    chrome_options = Options()
    # 注意：这里不使用无头模式，方便调试
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("正在访问长毛象网站...")
        driver.get("https://m.cmx.im/public/local")
        
        # 等待页面加载
        logger.info("等待页面加载...")
        time.sleep(10)  # 增加等待时间
        
        # 获取页面标题
        page_title = driver.title
        logger.info(f"页面标题: {page_title}")
        
        # 检查页面是否包含应用容器
        try:
            app_container = driver.find_element(By.ID, "mastodon")
            logger.info("✓ 找到Mastodon应用容器")
        except:
            logger.error("✗ 未找到Mastodon应用容器")
        
        # 等待更长时间让JavaScript完全加载
        logger.info("等待JavaScript内容加载...")
        time.sleep(15)
        
        # 尝试多种可能的选择器
        selectors_to_test = [
            # 标准Mastodon选择器
            "article",
            ".status",
            "[data-testid='status']",
            ".status__wrapper",
            ".detailed-status",
            
            # 通用选择器
            "div[role='article']",
            "[role='listitem']",
            ".timeline-item",
            
            # 更宽泛的选择器
            "div[class*='status']",
            "div[class*='post']",
            "div[class*='toot']",
            
            # React组件相关
            "div[data-react-class]",
            "div[data-component]",
        ]
        
        logger.info("测试各种CSS选择器...")
        found_elements = {}
        
        for selector in selectors_to_test:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    found_elements[selector] = len(elements)
                    logger.info(f"✓ {selector}: 找到 {len(elements)} 个元素")
                    
                    # 检查前几个元素的内容
                    for i, element in enumerate(elements[:3]):
                        try:
                            text = element.text.strip()
                            if text:
                                logger.info(f"  元素 {i+1} 内容预览: {text[:100]}...")
                        except:
                            pass
                else:
                    logger.info(f"✗ {selector}: 未找到元素")
            except Exception as e:
                logger.warning(f"✗ {selector}: 查找失败 - {e}")
        
        # 检查页面源码中的关键信息
        logger.info("分析页面源码...")
        page_source = driver.page_source
        
        # 检查是否包含时间线数据
        if "timeline" in page_source.lower():
            logger.info("✓ 页面包含timeline相关内容")
        else:
            logger.warning("✗ 页面不包含timeline相关内容")
        
        # 检查是否包含状态数据
        if "status" in page_source.lower():
            logger.info("✓ 页面包含status相关内容")
        else:
            logger.warning("✗ 页面不包含status相关内容")
        
        # 检查是否有错误信息
        if "error" in page_source.lower():
            logger.warning("⚠ 页面可能包含错误信息")
        
        # 保存页面源码用于调试
        with open("debug_page_source.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        logger.info("页面源码已保存到 debug_page_source.html")
        
        # 尝试执行JavaScript获取更多信息
        logger.info("尝试执行JavaScript获取页面信息...")
        try:
            # 检查React应用状态
            react_info = driver.execute_script("""
                return {
                    hasReact: typeof React !== 'undefined',
                    hasRedux: typeof window.store !== 'undefined',
                    bodyClasses: document.body.className,
                    appContainer: document.getElementById('mastodon') ? 'found' : 'not found',
                    visibleElements: document.querySelectorAll('*').length
                };
            """)
            logger.info(f"JavaScript信息: {react_info}")
        except Exception as e:
            logger.warning(f"JavaScript执行失败: {e}")
        
        # 截图保存
        try:
            driver.save_screenshot("debug_screenshot.png")
            logger.info("截图已保存到 debug_screenshot.png")
        except Exception as e:
            logger.warning(f"截图失败: {e}")
        
        # 总结
        logger.info("=" * 50)
        logger.info("调试总结:")
        logger.info(f"找到的元素选择器: {list(found_elements.keys())}")
        if found_elements:
            best_selector = max(found_elements.items(), key=lambda x: x[1])
            logger.info(f"推荐使用的选择器: {best_selector[0]} (找到 {best_selector[1]} 个元素)")
        else:
            logger.error("未找到任何帖子元素！可能需要:")
            logger.error("1. 增加等待时间")
            logger.error("2. 处理登录或验证")
            logger.error("3. 使用不同的访问方式")
        
        input("按Enter键关闭浏览器...")  # 暂停以便观察
        
    except Exception as e:
        logger.error(f"调试过程出错: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()


if __name__ == "__main__":
    debug_page_structure()