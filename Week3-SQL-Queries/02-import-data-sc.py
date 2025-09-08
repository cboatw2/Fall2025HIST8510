# Import SC CSV Data into Database
# History 8510 - Clemson University
# This script populates the SC database with data from sc-data.csv

import sqlite3
import csv

print("=== Importing South Carolina Gay Guides Data ===")

# Step 1: Connect to existing database
print("\nStep 1: Connecting to existing database...")
conn = sqlite3.connect('sc_gay_guides.db')
cursor = conn.cursor()
print("✓ Connected to sc_gay_guides.db")

# Step 2: Read the CSV file
print("\nStep 2: Reading CSV file...")
venues_data = []

try:
    with open('sc-data.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            venues_data.append(row)
    print(f"✓ Read {len(venues_data)} records from sc-data.csv")
except FileNotFoundError:
    print("❌ Error: sc-data.csv file not found!")
    print("Make sure the CSV file is in the same folder as this script")
    exit()

# Step 3: Import unique cities first
print("\nStep 3: Importing cities...")
cities_added = 0
cities_skipped = 0

# .strip removes leading/trailing whitespace
for venue in venues_data:
    city = venue['city'].strip() if venue['city'] else None
    state = venue['state'].strip() if venue['state'] else None
    
    if city and state:  # Make sure we have both city and state
        try:
            cursor.execute('''
            INSERT INTO cities (city_name, state)
            VALUES (?, ?)
            ''', (city, state))
            cities_added += 1
        except sqlite3.IntegrityError:
            # City already exists due to UNIQUE constraint
            cities_skipped += 1

print(f"✓ Added {cities_added} new cities")
print(f"✓ Skipped {cities_skipped} duplicate cities")

# Step 4: Import venues with foreign key relationships
print("\nStep 4: Importing venues...")
venues_added = 0
venues_failed = 0

for i, venue in enumerate(venues_data):
    try:
        # Get city_id by looking up the city/state combination
        city_id = None
        if venue['city'] and venue['state']:
            cursor.execute('''
            SELECT city_id FROM cities 
            WHERE city_name = ? AND state = ?
            ''', (venue['city'].strip(), venue['state'].strip()))
            result = cursor.fetchone()
            if result:
                city_id = result[0]
        
        # Handle year (convert to integer if possible)
        year = None
        if venue['Year']:
            try:
                year = int(float(venue['Year']))
            except (ValueError, TypeError):
                year = None
        
        # Insert venue with all available data
        cursor.execute('''
        INSERT INTO venues (
            unique_id, title, description, type, street_address,
            city_id, year, amenity_features, notes, full_address,
            lat, lon, geo_address, unclear_address, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            venue['unique.id'],
            venue['title'],
            venue['description'],
            venue['type'],
            venue['streetaddress'],
            city_id,
            year,
            venue['amenityfeatures'],
            venue['notes'],
            venue['full.address'],
            float(venue['lat']) if venue['lat'] and venue['lat'].strip() else None,
            float(venue['lon']) if venue['lon'] and venue['lon'].strip() else None,
            venue['geoAddress'],
            venue['unclear_address'],
            venue['status']
        ))
        venues_added += 1
        
    except Exception as e:
        venues_failed += 1
        print(f"   ⚠️  Failed to import venue {i+1}: {e}")

print(f"✓ Added {venues_added} venues")
if venues_failed > 0:
    print(f"❌ Failed to import {venues_failed} venues")

# Step 5: Commit changes and show summary
print("\nStep 5: Saving all data...")
conn.commit()

# Final database summary
print("\n=== Database Summary ===")
cursor.execute("SELECT COUNT(*) FROM cities")
city_count = cursor.fetchone()[0]
print(f"Cities: {city_count}")

cursor.execute("SELECT COUNT(*) FROM venues")
venue_count = cursor.fetchone()[0]
print(f"Venues: {venue_count}")

# Show some sample data
print("\n=== Sample Data ===")
print("\nCities:")
cursor.execute("SELECT city_name, state FROM cities ORDER BY city_name LIMIT 5")
for row in cursor.fetchall():
    print(f"   {row[0]}, {row[1]}")

print("\nVenues with locations:")
cursor.execute('''
SELECT v.title, c.city_name, c.state, v.year, v.type
FROM venues v
JOIN cities c ON v.city_id = c.city_id
LIMIT 5
''')
for row in cursor.fetchall():
    title = row[0] if row[0] else "No title"
    print(f"   {title} in {row[1]}, {row[2]} ({row[3]}) - {row[4]}")

# Show year distribution
print("\nYear Distribution:")
cursor.execute('''
SELECT year, COUNT(*) as count
FROM venues
WHERE year IS NOT NULL
GROUP BY year
ORDER BY year
''')
for row in cursor.fetchall():
    print(f"   {row[0]}: {row[1]} venues")

# Close connection
conn.close()
print("\n✓ Database import complete!")
print("✓ Database saved as sc_gay_guides.db")

