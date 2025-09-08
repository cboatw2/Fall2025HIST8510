#!/usr/bin/env python3
"""
Simple Geographic Analysis of LGBTQ+ Venues in South Carolina
History 8510 - Clemson University

This script answers the question: How did LGBTQ+ venue coverage expand 
geographically across South Carolina over time?

Uses a simple SQL query to analyze venue distribution by city and year.
"""

import sqlite3
import pandas as pd

def analyze_geographic_expansion():
    """Analyze how LGBTQ+ venue coverage expanded geographically over time in SC"""
    
    print("=== LGBTQ+ Venue Geographic Expansion Analysis - South Carolina ===")
    print("History 8510 - Clemson University")
    print("=" * 60)
    
    # Connect to SC database
    try:
        conn = sqlite3.connect('sc_gay_guides.db')
        print("✓ Connected to SC database (sc_gay_guides.db)")
    except sqlite3.Error as e:
        print(f"❌ Database connection error: {e}")
        return
    
    # Core query for SC database (simplified - no publications table needed)
    query = """
    SELECT 
        c.city_name,
        c.state,
        v.year,
        COUNT(*) as venue_count
    FROM venues v
    JOIN cities c ON v.city_id = c.city_id
    WHERE v.year IS NOT NULL
    GROUP BY c.city_name, c.state, v.year
    ORDER BY v.year, venue_count DESC
    """
    
    print("\n🔍 Executing geographic expansion query...")
    
    try:
        # Execute query and load into pandas
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("❌ No data returned from query")
            return
        
        print(f"✓ Query successful! Retrieved {len(df)} records")
        
        # Basic analysis
        print(f"\n📊 Analysis Summary:")
        print(f"   Total cities covered: {df['city_name'].nunique()}")
        print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"   Total venue records: {df['venue_count'].sum()}")
        
        # Show expansion over time
        print(f"\n🌍 Geographic Expansion Over Time:")
        yearly_summary = df.groupby('year').agg({
            'city_name': 'nunique',
            'venue_count': 'sum'
        }).rename(columns={'city_name': 'cities_covered', 'venue_count': 'total_venues'})
        
        for year, row in yearly_summary.iterrows():
            print(f"   {year}: {row['cities_covered']} cities, {row['total_venues']} venues")
        
        # Show top cities by total venues
        print(f"\n🏙️  Top Cities by Total Venues:")
        city_totals = df.groupby(['city_name', 'state'])['venue_count'].sum().sort_values(ascending=False)
        for (city, state), count in city_totals.head(10).items():
            print(f"   {city}, {state}: {count} venues")
        
        # Show new cities appearing each year
        print(f"\n🆕 New Cities Added Each Year:")
        cities_by_year = {}
        for _, row in df.iterrows():
            year = row['year']
            city_state = f"{row['city_name']}, {row['state']}"
            if year not in cities_by_year:
                cities_by_year[year] = set()
            cities_by_year[year].add(city_state)
        
        all_cities = set()
        for year in sorted(cities_by_year.keys()):
            new_cities = cities_by_year[year] - all_cities
            if new_cities:
                print(f"   {year}: {', '.join(sorted(new_cities))}")
                all_cities.update(new_cities)
        
        # Export results
        output_file = "sc_geographic_expansion_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results exported to: {output_file}")
        
        # Show sample of the data
        print(f"\n📋 Sample Data (first 10 rows):")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error executing query: {e}")
    
    finally:
        conn.close()
        print("\n✓ Database connection closed")

if __name__ == "__main__":
    analyze_geographic_expansion()
