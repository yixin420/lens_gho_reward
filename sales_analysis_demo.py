#!/usr/bin/env python3
"""
Sales Data Analysis Demo
This script demonstrates the analysis that would be performed on your sales data.
Since we don't have access to the actual data file, this creates sample data
that matches typical sales data structure and performs all requested analyses.
"""

import json
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
import random

def create_sample_data():
    """
    Create sample sales data that matches typical structure
    """
    print("Creating sample sales data to demonstrate analysis...")
    
    # Sample data structure
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    
    data = []
    base_date = datetime(2023, 1, 1)
    
    # Generate 2000 sample records
    for i in range(2000):
        # Random date between 2023-2024
        days_offset = random.randint(0, 730)  # 2 years
        current_date = base_date + timedelta(days=days_offset)
        
        region = random.choice(regions)
        product = random.choice(products)
        
        # Base prices with some variation
        base_prices = {
            'Product A': 25.0,
            'Product B': 15.0,
            'Product C': 35.0,
            'Product D': 20.0,
            'Product E': 45.0
        }
        
        # Add regional and seasonal variations
        price_multiplier = 1.0
        if region == 'East':
            price_multiplier *= 1.1  # Higher prices in East
        elif region == 'South':
            price_multiplier *= 0.95  # Lower prices in South
        
        # Seasonal variations
        if current_date.month in [11, 12]:  # Holiday season
            price_multiplier *= 1.15
        elif current_date.month in [1, 2]:  # Post-holiday
            price_multiplier *= 0.9
        
        unit_price = base_prices[product] * price_multiplier * random.uniform(0.8, 1.2)
        units_sold = random.randint(1, 50)
        
        # Occasional outliers
        if random.random() < 0.05:  # 5% chance of outlier
            unit_price *= random.uniform(0.3, 3.0)  # Extreme price variation
        
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'region': region,
            'product': product,
            'unit_price': round(unit_price, 2),
            'units_sold': units_sold,
            'revenue': round(unit_price * units_sold, 2)
        })
    
    return data

def analyze_data(data):
    """
    Perform comprehensive analysis on the sales data
    """
    print("=" * 80)
    print("ADVANCED SALES DATA ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"Dataset contains {len(data)} transactions")
    
    # Convert date strings to datetime objects and add temporal features
    for record in data:
        date_obj = datetime.strptime(record['date'], '%Y-%m-%d')
        record['year'] = date_obj.year
        record['month'] = date_obj.month
        record['quarter'] = (date_obj.month - 1) // 3 + 1
        record['month_name'] = date_obj.strftime('%B')
    
    # TASK 3: Group by month, region, product and analyze pricing
    print("\n\n3. GROUPING ANALYSIS AND ANOMALY DETECTION")
    print("-" * 60)
    
    # Group data
    groups = defaultdict(list)
    for record in data:
        key = (record['month'], record['region'], record['product'])
        groups[key].append(record['unit_price'])
    
    # Calculate group statistics
    group_stats = []
    overall_prices = [record['unit_price'] for record in data]
    overall_mean = statistics.mean(overall_prices)
    overall_stdev = statistics.stdev(overall_prices)
    
    print(f"Created {len(groups)} unique month-region-product combinations")
    print(f"Overall average unit price: ${overall_mean:.2f}")
    print(f"Overall standard deviation: ${overall_stdev:.2f}")
    
    for (month, region, product), prices in groups.items():
        if len(prices) >= 3:  # Minimum sample size for reliability
            avg_price = statistics.mean(prices)
            z_score = (avg_price - overall_mean) / overall_stdev
            
            group_stats.append({
                'month': month,
                'region': region,
                'product': product,
                'avg_price': avg_price,
                'count': len(prices),
                'z_score': z_score
            })
    
    # Define anomaly criteria
    print("\nANOMALY DETECTION METHODOLOGY:")
    print("1. Statistical approach: Using Z-score (>2.5 or <-2.5) to identify extreme values")
    print("2. Business context: Considering sample size (min 3 transactions) for reliability")
    print("3. Relative comparison: Identifying prices significantly different from overall patterns")
    
    # Find unusual groups
    unusual_high = [g for g in group_stats if g['z_score'] > 2.5]
    unusual_low = [g for g in group_stats if g['z_score'] < -2.5]
    
    print(f"\n🔴 UNUSUALLY HIGH PRICE GROUPS (Z-score > 2.5):")
    if unusual_high:
        unusual_high.sort(key=lambda x: x['z_score'], reverse=True)
        for group in unusual_high[:10]:  # Top 10
            print(f"  Month {group['month']:2d} | {group['region']:5} | {group['product']:10} | "
                  f"${group['avg_price']:7.2f} | Z-score: {group['z_score']:5.2f} | "
                  f"Transactions: {group['count']}")
    else:
        print("  No unusually high price groups found")
    
    print(f"\n🔵 UNUSUALLY LOW PRICE GROUPS (Z-score < -2.5):")
    if unusual_low:
        unusual_low.sort(key=lambda x: x['z_score'])
        for group in unusual_low[:10]:  # Bottom 10
            print(f"  Month {group['month']:2d} | {group['region']:5} | {group['product']:10} | "
                  f"${group['avg_price']:7.2f} | Z-score: {group['z_score']:5.2f} | "
                  f"Transactions: {group['count']}")
    else:
        print("  No unusually low price groups found")
    
    print(f"\nPOSSIBLE CAUSES FOR PRICE ANOMALIES:")
    print("• High prices: Premium product variants, seasonal demand, limited supply, market positioning")
    print("• Low prices: Promotional campaigns, bulk discounts, market penetration strategy, clearance sales")
    print("• Regional factors: Local competition, economic conditions, distribution costs, market maturity")
    print("• Temporal factors: Holiday seasons, end-of-quarter sales, inventory management cycles")
    
    # TASK 4: South region 2024 analysis
    print("\n\n4. SOUTH REGION 2024 ANALYSIS")
    print("-" * 40)
    
    # Filter for South region 2024
    south_2024 = [r for r in data if r['region'] == 'South' and r['year'] == 2024]
    
    if south_2024:
        print(f"South region 2024 data: {len(south_2024)} transactions")
        
        # Group by product and sum units sold
        product_sales = defaultdict(int)
        for record in south_2024:
            product_sales[record['product']] += record['units_sold']
        
        # Sort by sales volume
        sorted_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Products analyzed: {len(sorted_products)}")
        
        # Create text-based bar chart (since matplotlib isn't available)
        print(f"\nSOUTH REGION 2024: SALES VOLUMES BY PRODUCT")
        print("-" * 50)
        
        max_sales = max(product_sales.values()) if product_sales else 1
        total_sales = sum(product_sales.values())
        
        for i, (product, sales) in enumerate(sorted_products, 1):
            bar_length = int((sales / max_sales) * 40)  # Scale to 40 characters
            bar = "█" * bar_length
            market_share = (sales / total_sales) * 100
            
            print(f"{i}. {product:12} │{bar:<40}│ {sales:,} units ({market_share:.1f}%)")
        
        # Business observations
        print(f"\nBUSINESS OBSERVATIONS:")
        if sorted_products:
            top_product, top_sales = sorted_products[0]
            top_market_share = (top_sales / total_sales) * 100
            
            print(f"1. Market Leadership: '{top_product}' dominates with {top_sales:,} units ({top_market_share:.1f}% market share)")
            print(f"2. Sales Distribution: Total units sold = {total_sales:,} across {len(sorted_products)} products")
            
            if len(sorted_products) > 1:
                second_product, second_sales = sorted_products[1]
                performance_gap = ((top_sales - second_sales) / second_sales) * 100
                print(f"3. Performance Gap: {performance_gap:.1f}% difference between top two products")
        
        print(f"\nSTRATEGIC RECOMMENDATIONS:")
        print(f"• Focus Resources: Invest more in '{sorted_products[0][0]}' supply chain and marketing")
        print(f"• Portfolio Balance: Consider promoting underperforming products or discontinuing weak ones")
        print(f"• Market Expansion: Leverage '{sorted_products[0][0]}' success to enter adjacent markets")
        print(f"• Inventory Management: Optimize stock levels based on demonstrated demand patterns")
    else:
        print("No South region data found for 2024")
    
    # TASK 5: Financial performance analysis
    print("\n\n5. FINANCIAL PERFORMANCE ANALYSIS")
    print("-" * 45)
    
    # 5(a): East region revenue growth 2023-2024
    print("5(a) EAST REGION REVENUE GROWTH (2023-2024)")
    print("-" * 50)
    
    east_data = [r for r in data if r['region'] == 'East']
    
    if east_data:
        revenue_2023 = sum(r['revenue'] for r in east_data if r['year'] == 2023)
        revenue_2024 = sum(r['revenue'] for r in east_data if r['year'] == 2024)
        
        if revenue_2023 > 0 and revenue_2024 > 0:
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
    
    south_product_a = [r for r in data if r['region'] == 'South' and r['product'] == 'Product A']
    
    if south_product_a:
        q1_sales = sum(r['units_sold'] for r in south_product_a if r['quarter'] == 1)
        q4_sales = sum(r['units_sold'] for r in south_product_a if r['quarter'] == 4)
        
        if q1_sales > 0 and q4_sales > 0:
            q4_vs_q1_change = ((q4_sales - q1_sales) / q1_sales) * 100
            
            print(f"Product A Sales in South Region:")
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
        print("No Product A data found in South region")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("Note: This analysis used sample data to demonstrate the methodology.")
    print("Replace the sample data creation with actual CSV loading for real analysis.")
    print("=" * 80)

def main():
    """
    Main function to run the complete analysis
    """
    print("SALES DATA ANALYSIS DEMONSTRATION")
    print("=" * 50)
    print("This script demonstrates the analysis methodology using sample data.")
    print("In practice, this would load your actual CSV file from:")
    print("C:\\Users\\Lenovo\\Desktop\\finish study\\sales_data.csv")
    print()
    
    # Create and analyze sample data
    sample_data = create_sample_data()
    analyze_data(sample_data)
    
    print("\nTo use with your actual data:")
    print("1. Install required packages: pip install pandas numpy matplotlib seaborn")
    print("2. Run: python advanced_sales_analysis.py")
    print("3. The script will automatically detect your CSV columns and perform the analysis")

if __name__ == "__main__":
    main()