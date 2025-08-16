# Script 2: Import CSV Data into Database
# History 8510 - August 25
# This script populates our database with data from the CSV file

import sqlite3
import csv

print("=== Importing Gay Guides Data ===")

# Step 1: Connect to existing database
print("\nStep 1: Connecting to existing database...")
conn = sqlite3.connect('gay_guides.db')
cursor = conn.cursor()
print("✓ Connected to gay_guides.db")

# Step 2: Read the CSV file
print("\nStep 2: Reading CSV file...")
venues_data = []

try:
    with open('corand100.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            venues_data.append(row)
    print(f"✓ Read {len(venues_data)} records from corand100.csv")
except FileNotFoundError:
    print("❌ Error: corand100.csv file not found!")
    print("Make sure the CSV file is in the same folder as this script")
    exit()

# Step 3: Import unique cities first
print("\nStep 3: Importing cities...")
cities_added = 0
cities_skipped = 0

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

# Step 4: Import unique publications
print("\nStep 4: Importing publications...")
publications_added = 0
publications_skipped = 0

for venue in venues_data:
    publication = venue['publication'].strip() if venue['publication'] else None
    year_str = venue['year']
    
    if publication and year_str:
        try:
            year = int(float(year_str))  # Handle potential decimal years
            cursor.execute('''
            INSERT INTO publications (publication_name, year)
            VALUES (?, ?)
            ''', (publication, year))
            publications_added += 1
        except (ValueError, sqlite3.IntegrityError):
            # Invalid year or duplicate publication
            publications_skipped += 1

print(f"✓ Added {publications_added} new publications")
print(f"✓ Skipped {publications_skipped} duplicate publications")

# Step 5: Import venues with foreign key relationships
print("\nStep 5: Importing venues...")
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
        
        # Get pub_id by looking up publication/year combination
        pub_id = None
        if venue['publication'] and venue['year']:
            try:
                year = int(float(venue['year']))
                cursor.execute('''
                SELECT pub_id FROM publications 
                WHERE publication_name = ? AND year = ?
                ''', (venue['publication'].strip(), year))
                result = cursor.fetchone()
                if result:
                    pub_id = result[0]
            except ValueError:
                pass  # Invalid year
        
        # Insert venue with foreign key relationships
        cursor.execute('''
        INSERT INTO venues (
            record_id, title, description, type, address,
            city_id, pub_id, stars, star_type, amenity_features,
            status, geo_address, lat, lon
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            venue['record_id'],
            venue['title'],
            venue['description'],
            venue['type'],
            venue['address'],
            city_id,
            pub_id,
            venue['stars'],
            venue['star.type'],
            venue['amenityfeatures'],
            venue['status'],
            venue['geoAddress'],
            float(venue['lat']) if venue['lat'] and venue['lat'].strip() else None,
            float(venue['lon']) if venue['lon'] and venue['lon'].strip() else None
        ))
        venues_added += 1
        
    except Exception as e:
        venues_failed += 1
        print(f"   ⚠️  Failed to import venue {i+1}: {e}")

print(f"✓ Added {venues_added} venues")
if venues_failed > 0:
    print(f"❌ Failed to import {venues_failed} venues")

# Step 6: Commit changes and show summary
print("\nStep 6: Saving all data...")
conn.commit()

# Final database summary
print("\n=== Database Summary ===")
cursor.execute("SELECT COUNT(*) FROM cities")
city_count = cursor.fetchone()[0]
print(f"Cities: {city_count}")

cursor.execute("SELECT COUNT(*) FROM publications")
pub_count = cursor.fetchone()[0]
print(f"Publications: {pub_count}")

cursor.execute("SELECT COUNT(*) FROM venues")
venue_count = cursor.fetchone()[0]
print(f"Venues: {venue_count}")

# Show some sample data
print("\n=== Sample Data ===")
print("\nCities:")
cursor.execute("SELECT city_name, state FROM cities ORDER BY city_name LIMIT 5")
for row in cursor.fetchall():
    print(f"   {row[0]}, {row[1]}")

print("\nPublications:")
cursor.execute("SELECT publication_name, year FROM publications ORDER BY year LIMIT 3")
for row in cursor.fetchall():
    print(f"   {row[0]} ({row[1]})")

print("\nVenues with locations:")
cursor.execute('''
SELECT v.title, c.city_name, c.state, v.type
FROM venues v
JOIN cities c ON v.city_id = c.city_id
ORDER BY c.city_name
LIMIT 5
''')
for row in cursor.fetchall():
    print(f"   {row[0]} in {row[1]}, {row[2]} ({row[3]})")

# Close connection
conn.close()
print("\n=== Data import complete! ===")
print("Open gay_guides.db in VS Code SQLite viewer to explore the populated database")