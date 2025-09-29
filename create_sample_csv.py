#!/usr/bin/env python3
"""
Create a sample CSV file that matches your sales data structure
This allows you to test the analysis scripts with realistic data
"""

import csv
import random
from datetime import datetime, timedelta

def create_sample_csv():
    """Create a sample sales_data.csv file with realistic structure"""
    
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    
    # Create sample data
    data = []
    base_date = datetime(2023, 1, 1)
    
    print("Creating sample sales_data.csv...")
    
    for i in range(2000):
        # Random date between 2023-2024
        days_offset = random.randint(0, 730)
        current_date = base_date + timedelta(days=days_offset)
        
        region = random.choice(regions)
        product = random.choice(products)
        
        # Base prices with realistic variations
        base_prices = {
            'Product A': 25.0,
            'Product B': 15.0,
            'Product C': 35.0,
            'Product D': 20.0,
            'Product E': 45.0
        }
        
        # Regional pricing variations
        price_multiplier = 1.0
        if region == 'East':
            price_multiplier *= 1.1
        elif region == 'South':
            price_multiplier *= 0.95
        
        # Seasonal variations
        if current_date.month in [11, 12]:  # Holiday season
            price_multiplier *= 1.15
        elif current_date.month in [1, 2]:  # Post-holiday
            price_multiplier *= 0.9
        
        unit_price = base_prices[product] * price_multiplier * random.uniform(0.8, 1.2)
        units_sold = random.randint(1, 50)
        
        # Create some price anomalies (5% chance)
        if random.random() < 0.05:
            if random.random() < 0.5:
                unit_price *= random.uniform(2.0, 3.0)  # High price anomaly
            else:
                unit_price *= random.uniform(0.3, 0.6)  # Low price anomaly
        
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'region': region,
            'product': product,
            'unit_price': round(unit_price, 2),
            'units_sold': units_sold,
            'revenue': round(unit_price * units_sold, 2)
        })
    
    # Write to CSV
    with open('/workspace/sample_sales_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'region', 'product', 'unit_price', 'units_sold', 'revenue']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    print(f"✓ Created sample_sales_data.csv with {len(data)} records")
    print("Columns: date, region, product, unit_price, units_sold, revenue")
    print("\nThis sample file demonstrates the expected data structure.")
    print("You can use this to test the analysis scripts before running on your actual data.")
    
    # Show sample of the data
    print("\nSample data preview:")
    print("-" * 80)
    print("Date       | Region | Product   | Unit Price | Units | Revenue")
    print("-" * 80)
    for i in range(10):
        row = data[i]
        print(f"{row['date']} | {row['region']:6} | {row['product']:9} | "
              f"${row['unit_price']:8.2f} | {row['units_sold']:5} | ${row['revenue']:8.2f}")

if __name__ == "__main__":
    create_sample_csv()