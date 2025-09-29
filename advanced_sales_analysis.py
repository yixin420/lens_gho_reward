import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    Advanced Sales Data Analysis Script
    
    Design Thoughts and Approach:
    1. Data preprocessing strategy: We'll first ensure proper date parsing and data type conversion
       to enable accurate temporal analysis across months, quarters, and years.
    2. Anomaly detection methodology: We'll use statistical methods (Z-score and IQR) to identify
       unusual pricing patterns, considering that business context matters more than pure statistics.
    3. Visualization and business intelligence: Charts will be designed to reveal actionable insights
       rather than just displaying data, with focus on trends that impact business decisions.
    """
    
    # File path configuration
    file_path = r"C:\Users\Lenovo\Desktop\finish study\sales_data.csv"
    
    print("=" * 80)
    print("ADVANCED SALES DATA ANALYSIS REPORT")
    print("=" * 80)
    
    try:
        # Load data with multiple encoding attempts
        df = None
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✓ Successfully loaded data using {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"✗ File not found: {file_path}")
                return
        
        if df is None:
            print("✗ Unable to read file. Please check file format and encoding.")
            return
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Data preprocessing and column identification
        print("\nDATA PREPROCESSING")
        print("-" * 50)
        
        # Identify key columns automatically
        date_col = identify_column(df, ['date', 'Date', 'order_date', 'sale_date', 'time', '日期'])
        region_col = identify_column(df, ['region', 'Region', 'area', 'location', '地区'])
        product_col = identify_column(df, ['product', 'Product', 'product_name', 'item', '产品'])
        price_col = identify_column(df, ['price', 'unit_price', 'Price', 'cost', '单价'])
        quantity_col = identify_column(df, ['quantity', 'units_sold', 'qty', 'amount', '数量'])
        
        print(f"Identified columns:")
        print(f"  Date: {date_col}")
        print(f"  Region: {region_col}")
        print(f"  Product: {product_col}")
        print(f"  Unit Price: {price_col}")
        print(f"  Quantity/Units Sold: {quantity_col}")
        
        if not all([date_col, region_col, product_col, price_col]):
            print("\n⚠️  Some required columns could not be identified automatically.")
            print("Available columns:", list(df.columns))
            print("Please manually specify column names in the script.")
            return
        
        # Convert date column and extract temporal features
        df[date_col] = pd.to_datetime(df[date_col])
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['month_name'] = df[date_col].dt.strftime('%B')
        
        # Calculate revenue if quantity column exists
        if quantity_col:
            df['revenue'] = df[price_col] * df[quantity_col]
        
        # TASK 3: Group analysis and anomaly detection
        print("\n\n3. GROUPING ANALYSIS AND ANOMALY DETECTION")
        print("-" * 60)
        
        # Group by month, region, and product
        grouped = df.groupby(['month', region_col, product_col])[price_col].agg(['mean', 'count', 'std']).reset_index()
        grouped.columns = ['month', 'region', 'product', 'avg_unit_price', 'count', 'price_std']
        
        print(f"Created {len(grouped)} unique month-region-product combinations")
        
        # Design Thought: Anomaly Detection Criteria
        print("\nANOMALY DETECTION METHODOLOGY:")
        print("1. Statistical approach: Using Z-score (>2.5 or <-2.5) to identify extreme values")
        print("2. Business context: Considering sample size (min 3 transactions) for reliability")
        print("3. Relative comparison: Identifying prices significantly different from overall patterns")
        
        # Calculate Z-scores for anomaly detection
        overall_mean = df[price_col].mean()
        overall_std = df[price_col].std()
        grouped['z_score'] = (grouped['avg_unit_price'] - overall_mean) / overall_std
        
        # Define criteria for unusual groups
        unusual_high = grouped[(grouped['z_score'] > 2.5) & (grouped['count'] >= 3)]
        unusual_low = grouped[(grouped['z_score'] < -2.5) & (grouped['count'] >= 3)]
        
        print(f"\nOverall average unit price: ${overall_mean:.2f}")
        print(f"Standard deviation: ${overall_std:.2f}")
        
        print(f"\n🔴 UNUSUALLY HIGH PRICE GROUPS (Z-score > 2.5):")
        if len(unusual_high) > 0:
            for _, row in unusual_high.iterrows():
                print(f"  Month {int(row['month']):2d} | {row['region']:12} | {row['product']:15} | "
                      f"${row['avg_unit_price']:7.2f} | Z-score: {row['z_score']:5.2f} | "
                      f"Transactions: {int(row['count'])}")
        else:
            print("  No unusually high price groups found")
        
        print(f"\n🔵 UNUSUALLY LOW PRICE GROUPS (Z-score < -2.5):")
        if len(unusual_low) > 0:
            for _, row in unusual_low.iterrows():
                print(f"  Month {int(row['month']):2d} | {row['region']:12} | {row['product']:15} | "
                      f"${row['avg_unit_price']:7.2f} | Z-score: {row['z_score']:5.2f} | "
                      f"Transactions: {int(row['count'])}")
        else:
            print("  No unusually low price groups found")
        
        # Analysis of possible causes
        print(f"\nPOSSIBLE CAUSES FOR PRICE ANOMALIES:")
        print("• High prices: Premium product variants, seasonal demand, limited supply, market positioning")
        print("• Low prices: Promotional campaigns, bulk discounts, market penetration strategy, clearance sales")
        print("• Regional factors: Local competition, economic conditions, distribution costs, market maturity")
        print("• Temporal factors: Holiday seasons, end-of-quarter sales, inventory management cycles")
        
        # TASK 4: South region 2024 visualization and analysis
        print("\n\n4. SOUTH REGION 2024 ANALYSIS")
        print("-" * 40)
        
        if quantity_col:
            # Filter for South region 2024 data
            south_2024 = df[(df[region_col].str.contains('South', case=False, na=False)) & 
                           (df['year'] == 2024)]
            
            if len(south_2024) > 0:
                # Group by product and sum units sold
                product_sales = south_2024.groupby(product_col)[quantity_col].sum().sort_values(ascending=False)
                
                print(f"South region 2024 sales data: {len(south_2024)} transactions")
                print(f"Products analyzed: {len(product_sales)}")
                
                # Create bar chart
                plt.figure(figsize=(12, 8))
                bars = plt.bar(range(len(product_sales)), product_sales.values, 
                              color=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#708090'][:len(product_sales)])
                
                plt.title('South Region 2024: Sales Volumes by Product', fontsize=16, fontweight='bold')
                plt.xlabel('Products', fontsize=12)
                plt.ylabel('Units Sold', fontsize=12)
                plt.xticks(range(len(product_sales)), product_sales.index, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('/workspace/south_region_2024_sales.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Business observations and recommendations
                print(f"\nBUSINESS OBSERVATIONS:")
                top_product = product_sales.index[0]
                top_sales = product_sales.iloc[0]
                total_sales = product_sales.sum()
                market_share = (top_sales / total_sales) * 100
                
                print(f"1. Market Leadership: '{top_product}' dominates with {top_sales:,} units ({market_share:.1f}% market share)")
                print(f"2. Sales Distribution: Total units sold = {total_sales:,} across {len(product_sales)} products")
                
                if len(product_sales) > 1:
                    second_product = product_sales.index[1]
                    performance_gap = ((product_sales.iloc[0] - product_sales.iloc[1]) / product_sales.iloc[1]) * 100
                    print(f"3. Performance Gap: {performance_gap:.1f}% difference between top two products")
                
                print(f"\nSTRATEGIC RECOMMENDATIONS:")
                print(f"• Focus Resources: Invest more in '{top_product}' supply chain and marketing")
                print(f"• Portfolio Balance: Consider promoting underperforming products or discontinuing weak ones")
                print(f"• Market Expansion: Leverage '{top_product}' success to enter adjacent markets")
                print(f"• Inventory Management: Optimize stock levels based on demonstrated demand patterns")
                
            else:
                print("No South region data found for 2024")
        else:
            print("Quantity/Units sold column not identified - cannot create sales volume chart")
        
        # TASK 5: Revenue growth and quarterly analysis
        print("\n\n5. FINANCIAL PERFORMANCE ANALYSIS")
        print("-" * 45)
        
        if quantity_col:
            # 5(a): East region revenue growth 2023-2024
            print("5(a) EAST REGION REVENUE GROWTH (2023-2024)")
            print("-" * 50)
            
            east_data = df[df[region_col].str.contains('East', case=False, na=False)]
            
            if len(east_data) > 0:
                yearly_revenue = east_data.groupby('year')['revenue'].sum()
                
                if 2023 in yearly_revenue.index and 2024 in yearly_revenue.index:
                    revenue_2023 = yearly_revenue[2023]
                    revenue_2024 = yearly_revenue[2024]
                    growth_rate = ((revenue_2024 - revenue_2023) / revenue_2023) * 100
                    
                    print(f"East Region Revenue:")
                    print(f"  2023: ${revenue_2023:,.2f}")
                    print(f"  2024: ${revenue_2024:,.2f}")
                    print(f"  Growth Rate: {growth_rate:+.2f}%")
                    
                    if growth_rate > 0:
                        print(f"✓ Positive growth indicates successful market expansion or pricing strategy")
                    else:
                        print(f"⚠️ Negative growth suggests market challenges or competitive pressure")
                else:
                    print("Insufficient data for 2023-2024 comparison")
            else:
                print("No East region data found")
            
            # 5(b): Product A quarterly comparison in South region
            print(f"\n5(b) PRODUCT A QUARTERLY ANALYSIS - SOUTH REGION")
            print("-" * 55)
            
            # Try to identify Product A (use first product if 'Product A' not found)
            product_a = None
            if 'Product A' in df[product_col].values:
                product_a = 'Product A'
            else:
                # Use the first product as proxy for Product A
                product_a = df[product_col].iloc[0]
                print(f"Note: Using '{product_a}' as Product A proxy")
            
            south_product_a = df[(df[region_col].str.contains('South', case=False, na=False)) & 
                               (df[product_col] == product_a)]
            
            if len(south_product_a) > 0:
                quarterly_sales = south_product_a.groupby('quarter')[quantity_col].sum()
                
                if 1 in quarterly_sales.index and 4 in quarterly_sales.index:
                    q1_sales = quarterly_sales[1]
                    q4_sales = quarterly_sales[4]
                    q4_vs_q1_change = ((q4_sales - q1_sales) / q1_sales) * 100
                    
                    print(f"Product A ({product_a}) Sales in South Region:")
                    print(f"  Q1 Sales: {q1_sales:,} units")
                    print(f"  Q4 Sales: {q4_sales:,} units")
                    print(f"  Q4 vs Q1 Change: {q4_vs_q1_change:+.1f}%")
                    
                    print(f"\nQ4 PERFORMANCE ANALYSIS:")
                    if q4_vs_q1_change > 10:
                        print("🚀 Strong Q4 Performance - Excellent seasonal momentum")
                        print("Hypothesis: Holiday season demand, effective marketing campaigns, or successful product positioning")
                    elif q4_vs_q1_change > 0:
                        print("📈 Moderate Q4 Growth - Steady improvement")
                        print("Hypothesis: Gradual market acceptance, word-of-mouth growth, or seasonal factors")
                    elif q4_vs_q1_change > -10:
                        print("📊 Stable Q4 Performance - Consistent demand")
                        print("Hypothesis: Mature product lifecycle, stable customer base, or market saturation")
                    else:
                        print("📉 Declining Q4 Performance - Requires attention")
                        print("Hypothesis: Increased competition, market saturation, or seasonal downturn")
                    
                    print(f"\nSTRATEGIC IMPLICATIONS:")
                    print("• Seasonal Planning: Use Q4 patterns for next year's inventory and marketing planning")
                    print("• Resource Allocation: Adjust Q1 strategies based on Q4 performance trends")
                    print("• Market Intelligence: Analyze competitor actions during Q4 transition period")
                else:
                    print("Insufficient quarterly data for comparison")
            else:
                print(f"No data found for Product A in South region")
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

def identify_column(df, possible_names):
    """
    Helper function to identify column names automatically
    """
    for name in possible_names:
        if name in df.columns:
            return name
    return None

if __name__ == "__main__":
    main()