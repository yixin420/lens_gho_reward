#!/usr/bin/env python3
"""
快速测试改进版爬虫的基本功能
"""

import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_test():
    """快速测试页面加载和元素查找"""
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("访问长毛象网站...")
        driver.get("https://m.cmx.im/public/local")
        
        # 等待页面加载
        time.sleep(10)
        
        # 检查页面标题
        title = driver.title
        logger.info(f"页面标题: {title}")
        
        # 检查应用容器
        try:
            app_container = driver.find_element(By.ID, "mastodon")
            logger.info("✓ 找到Mastodon应用容器")
        except:
            logger.error("✗ 未找到Mastodon应用容器")
        
        # 等待更长时间
        logger.info("等待JavaScript内容加载...")
        time.sleep(15)
        
        # 测试各种选择器
        test_selectors = [
            "article",
            ".status",
            "div[role='article']",
            "div[class*='status']",
            "[data-testid]",
            "main div > div"
        ]
        
        found_any = False
        for selector in test_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.info(f"✓ {selector}: 找到 {len(elements)} 个元素")
                    
                    # 检查前几个元素的内容
                    for i, element in enumerate(elements[:2]):
                        try:
                            text = element.text.strip()
                            if text and len(text) > 20:
                                logger.info(f"  元素 {i+1}: {text[:100]}...")
                                found_any = True
                        except:
                            pass
                else:
                    logger.info(f"✗ {selector}: 未找到元素")
            except Exception as e:
                logger.warning(f"✗ {selector}: 查找失败 - {e}")
        
        if not found_any:
            logger.warning("未找到任何有内容的元素")
            
            # 尝试滚动
            logger.info("尝试滚动页面...")
            for i in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                # 再次检查
                elements = driver.find_elements(By.CSS_SELECTOR, "div")
                text_elements = [el for el in elements if el.text.strip() and len(el.text.strip()) > 20]
                if text_elements:
                    logger.info(f"滚动后找到 {len(text_elements)} 个有文本的元素")
                    for j, el in enumerate(text_elements[:2]):
                        logger.info(f"  元素 {j+1}: {el.text.strip()[:100]}...")
                    break
        
        # 保存页面源码
        with open("quick_test_page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logger.info("页面源码已保存到 quick_test_page_source.html")
        
        driver.quit()
        
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    quick_test()