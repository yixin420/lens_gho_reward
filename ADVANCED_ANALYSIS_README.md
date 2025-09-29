# Advanced Sales Data Analysis

## Overview
This script performs comprehensive sales data analysis including anomaly detection, visualization, and business intelligence insights.

## Design Thoughts and Methodology

### 1. Data Preprocessing Strategy
- **Multi-encoding support**: Automatically tries different encodings (UTF-8, GBK, GB2312, Latin-1) for robust file reading
- **Intelligent column detection**: Uses pattern matching to identify key columns automatically
- **Temporal feature engineering**: Extracts year, month, quarter information for time-based analysis

### 2. Anomaly Detection Approach
- **Statistical foundation**: Uses Z-score methodology (threshold ±2.5) to identify outliers
- **Business context consideration**: Requires minimum 3 transactions per group for statistical reliability
- **Relative comparison**: Compares group averages against overall market patterns

### 3. Business Intelligence Focus
- **Actionable insights**: Analysis designed to support business decision-making
- **Visual storytelling**: Charts emphasize trends and patterns rather than raw data display
- **Strategic recommendations**: Each analysis includes specific business recommendations

## Key Features

### Task 3: Group Analysis & Anomaly Detection
- Groups data by month, region, and product
- Calculates average unit prices for each combination
- Identifies unusually high/low price groups using statistical methods
- Provides business context for price anomalies

**Criteria for "Unusual" Groups:**
- Z-score > 2.5 (unusually high) or < -2.5 (unusually low)
- Minimum 3 transactions for statistical reliability
- Comparison against overall market average

### Task 4: South Region 2024 Visualization
- Creates professional bar chart of sales volumes by product
- Includes detailed business observations
- Provides strategic recommendations based on market share analysis
- Saves chart as high-quality PNG file

### Task 5: Financial Performance Analysis
**5(a) East Region Revenue Growth:**
- Calculates year-over-year revenue growth (2023-2024)
- Provides interpretation of growth trends

**5(b) Quarterly Performance Analysis:**
- Compares Q4 vs Q1 performance for Product A in South region
- Includes hypothesis about performance drivers
- Offers strategic implications for planning

## Usage Instructions

### Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
python advanced_sales_analysis.py
```

### File Configuration
The script expects your data file at: `C:\Users\Lenovo\Desktop\finish study\sales_data.csv`

If your file is located elsewhere or has a different name, modify the `file_path` variable in the script.

## Expected Column Types
The script automatically detects columns with these naming patterns:

- **Date**: date, Date, order_date, sale_date, time, 日期
- **Region**: region, Region, area, location, 地区
- **Product**: product, Product, product_name, item, 产品
- **Price**: price, unit_price, Price, cost, 单价
- **Quantity**: quantity, units_sold, qty, amount, 数量

## Output Files
- **Console Report**: Comprehensive analysis results printed to terminal
- **Chart**: `south_region_2024_sales.png` - Bar chart of South region sales

## Business Value

### Anomaly Detection Benefits
- Identify pricing errors or opportunities
- Detect unusual market conditions
- Monitor competitive positioning
- Spot promotional effectiveness

### Visualization Insights
- Clear market share understanding
- Product performance ranking
- Visual trend identification
- Data-driven decision support

### Financial Analysis Value
- Revenue growth tracking
- Seasonal pattern recognition
- Regional performance comparison
- Strategic planning support

## Technical Notes
- Handles missing data gracefully
- Supports multiple date formats
- Robust error handling and user feedback
- Optimized for business reporting rather than technical analysis

## Example Output Interpretation

**High Price Anomalies might indicate:**
- Premium product positioning
- Supply chain constraints
- Seasonal demand spikes
- Market leadership opportunities

**Low Price Anomalies might suggest:**
- Promotional campaigns
- Competitive pressure
- Market penetration strategies
- Inventory clearance needs

This analysis framework provides both statistical rigor and business practicality for sales data insights.