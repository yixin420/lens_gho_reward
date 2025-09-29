# 销售数据分析脚本

## 功能说明
这个Python脚本用于分析销售数据CSV文件，完成以下任务：

1. **数据加载与预览**
   - 加载CSV文件并显示前5行和后5行
   - 显示数据集基本信息（行数、列数、列名）

2. **数据质量检查**
   - 检查缺失值并统计缺失比例
   - 检查数据类型
   - 检查异常值（如负值）
   - 检查重复行

3. **产品分析**
   - 计算不同产品的数量
   - 显示各产品出现次数
   - 分析各产品出现的日期范围

4. **地区收入分析**
   - 计算各地区总收入（销售量×单价）
   - 找出收入最高的地区
   - 详细说明分析方法

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

```bash
python sales_data_analysis.py
```

## 注意事项

1. **文件路径**: 脚本中默认的文件路径是 `C:\Users\Lenovo\Desktop\finish study\sales_data.csv`
   - 如果你的文件路径不同，请修改脚本中的 `file_path` 变量

2. **文件格式**: 脚本支持多种编码格式（UTF-8, GBK, GB2312, Latin-1）

3. **列名识别**: 脚本会自动尝试识别以下类型的列：
   - 产品列: product, Product, product_name, item, 产品, 商品等
   - 日期列: date, order_date, sale_date, 日期, 时间等  
   - 地区列: region, area, location, city, 地区, 区域等
   - 销售量列: quantity, qty, amount, 数量, 销售量等
   - 单价列: price, unit_price, cost, 单价, 价格等

4. **手动调整**: 如果自动识别失败，可以根据脚本输出的提示手动修改相应的列名变量

## 输出示例

脚本会生成详细的分析报告，包括：
- 数据概览
- 数据质量报告
- 产品统计信息
- 地区收入排名
- 分析方法说明