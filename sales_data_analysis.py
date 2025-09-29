import pandas as pd
import numpy as np
import os

def main():
    """
    销售数据分析脚本
    分析CSV文件中的销售数据，完成以下任务：
    1. 加载CSV文件并显示前5行和后5行
    2. 检查缺失或异常数据
    3. 计算不同产品数量及其出现日期
    4. 计算各地区总收入并找出最高的地区
    """
    
    # 文件路径 - 根据用户提供的信息
    file_path = r"C:\Users\Lenovo\Desktop\finish study\sales_data.csv"
    
    print("=" * 60)
    print("销售数据分析报告")
    print("=" * 60)
    
    try:
        # 任务1(a): 加载CSV文件并显示前5行和后5行
        print("\n1. 加载CSV文件并显示数据")
        print("-" * 40)
        
        # 尝试不同的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ 成功使用 {encoding} 编码加载文件")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"✗ 文件未找到: {file_path}")
                print("请确认文件路径和文件名是否正确")
                return
        
        if df is None:
            print("✗ 无法读取文件，请检查文件格式和编码")
            return
        
        print(f"\n数据集形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print(f"列名: {list(df.columns)}")
        
        print("\n前5行数据:")
        print(df.head())
        
        print("\n后5行数据:")
        print(df.tail())
        
        # 任务1(b): 检查缺失或异常数据
        print("\n\n2. 数据质量检查")
        print("-" * 40)
        
        # 检查缺失值
        missing_data = df.isnull().sum()
        print("缺失值统计:")
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} 个缺失值 ({percentage:.2f}%)")
        
        if missing_data.sum() == 0:
            print("  ✓ 没有发现缺失值")
        
        # 检查数据类型
        print("\n数据类型:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # 检查异常值（如果有数值列）
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            print("\n数值列统计信息:")
            print(df[numeric_columns].describe())
            
            # 检查负值（对于价格、数量等不应为负的列）
            for col in numeric_columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"  警告: {col} 列有 {negative_count} 个负值")
        
        # 检查重复行
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            print(f"\n重复行: {duplicate_count} 行")
        else:
            print("\n✓ 没有发现重复行")
        
        # 任务2: 产品分析
        print("\n\n3. 产品分析")
        print("-" * 40)
        
        # 尝试识别产品列（常见的列名）
        product_col = None
        possible_product_cols = ['product', 'Product', 'product_name', 'Product_Name', 
                               'item', 'Item', '产品', '商品']
        
        for col in possible_product_cols:
            if col in df.columns:
                product_col = col
                break
        
        if product_col is None:
            print("产品列候选:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")
            print("\n请手动修改脚本中的 product_col 变量来指定产品列")
            product_col = df.columns[0]  # 默认使用第一列
            print(f"当前使用列: {product_col}")
        
        # 计算不同产品数量
        unique_products = df[product_col].nunique()
        print(f"不同产品数量: {unique_products}")
        
        print("\n各产品出现次数:")
        product_counts = df[product_col].value_counts()
        print(product_counts)
        
        # 尝试识别日期列
        date_col = None
        possible_date_cols = ['date', 'Date', 'order_date', 'Order_Date', 
                            'sale_date', 'Sale_Date', '日期', '时间']
        
        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # 尝试通过数据类型识别日期列
            for col in df.columns:
                if df[col].dtype == 'object':
                    # 检查是否包含日期格式的字符串
                    sample_values = df[col].dropna().head().astype(str)
                    if any('-' in str(val) or '/' in str(val) for val in sample_values):
                        date_col = col
                        break
        
        if date_col:
            try:
                # 转换为日期类型
                df[date_col] = pd.to_datetime(df[date_col])
                
                print(f"\n各产品出现的日期范围 (基于 {date_col} 列):")
                for product in df[product_col].unique():
                    product_data = df[df[product_col] == product]
                    min_date = product_data[date_col].min()
                    max_date = product_data[date_col].max()
                    date_count = product_data[date_col].nunique()
                    print(f"  {product}: {min_date.date()} 到 {max_date.date()} ({date_count} 个不同日期)")
            except:
                print(f"无法解析日期列 {date_col}")
        else:
            print("未找到日期列，请手动指定")
        
        # 任务3: 地区收入分析
        print("\n\n4. 地区收入分析")
        print("-" * 40)
        
        # 尝试识别地区列
        region_col = None
        possible_region_cols = ['region', 'Region', 'area', 'Area', 'location', 'Location',
                              'city', 'City', '地区', '区域', '城市']
        
        for col in possible_region_cols:
            if col in df.columns:
                region_col = col
                break
        
        # 尝试识别销售量和单价列
        quantity_col = None
        price_col = None
        
        possible_quantity_cols = ['quantity', 'Quantity', 'qty', 'Qty', 'amount', 'Amount',
                                '数量', '销售量']
        possible_price_cols = ['price', 'Price', 'unit_price', 'Unit_Price', 'cost', 'Cost',
                             '单价', '价格']
        
        for col in possible_quantity_cols:
            if col in df.columns:
                quantity_col = col
                break
        
        for col in possible_price_cols:
            if col in df.columns:
                price_col = col
                break
        
        if region_col and quantity_col and price_col:
            # 计算收入
            df['revenue'] = df[quantity_col] * df[price_col]
            
            # 按地区计算总收入
            region_revenue = df.groupby(region_col)['revenue'].sum().sort_values(ascending=False)
            
            print("各地区总收入排名:")
            for i, (region, revenue) in enumerate(region_revenue.items(), 1):
                print(f"  {i}. {region}: ¥{revenue:,.2f}")
            
            highest_region = region_revenue.index[0]
            highest_revenue = region_revenue.iloc[0]
            
            print(f"\n总收入最高的地区: {highest_region}")
            print(f"总收入: ¥{highest_revenue:,.2f}")
            
            print(f"\n判断方法说明:")
            print(f"1. 使用列 '{quantity_col}' 作为销售量")
            print(f"2. 使用列 '{price_col}' 作为单价")
            print(f"3. 计算收入 = 销售量 × 单价")
            print(f"4. 按地区 '{region_col}' 分组并求和")
            print(f"5. 排序找出收入最高的地区")
            
        else:
            print("无法自动识别必要的列，请检查以下列是否存在:")
            print(f"  地区列: {region_col}")
            print(f"  销售量列: {quantity_col}")
            print(f"  单价列: {price_col}")
            print("\n当前数据列:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("请检查文件路径和格式是否正确")

if __name__ == "__main__":
    main()