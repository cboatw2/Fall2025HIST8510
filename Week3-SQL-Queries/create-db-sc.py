# Script 1: Creating SC Database Structure
# History 8510 - Clemson University
# This script creates the database tables for South Carolina data

import sqlite3

print("=== Creating South Carolina Gay Guides Database Structure ===")

# Step 1: Connect to database (creates file if it doesn't exist)
print("\nStep 1: Connecting to database...")
conn = sqlite3.connect('sc_gay_guides.db')
cursor = conn.cursor()
print("✓ Connected to sc_gay_guides.db")

# Step 2: Create cities table
print("\nStep 2: Creating cities table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS cities (
    city_id INTEGER PRIMARY KEY AUTOINCREMENT,
    city_name TEXT NOT NULL,
    state TEXT NOT NULL,
    UNIQUE(city_name, state)
)
''')
print("✓ Created cities table")
print("   - city_id: unique identifier (auto-generated)")
print("   - city_name: name of the city")
print("   - state: state abbreviation")
print("   - UNIQUE constraint: prevents duplicate city/state combinations")

# Step 3: Create venues table (simplified for SC data)
print("\nStep 3: Creating venues table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS venues (
    venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_id TEXT,
    title TEXT,
    description TEXT,
    type TEXT,
    street_address TEXT,
    city_id INTEGER,
    year INTEGER,
    amenity_features TEXT,
    notes TEXT,
    full_address TEXT,
    lat REAL,
    lon REAL,
    geo_address TEXT,
    unclear_address TEXT,
    status TEXT,
    FOREIGN KEY (city_id) REFERENCES cities (city_id)
)
''')
print("✓ Created venues table")
print("   - venue_id: unique identifier (auto-generated)")
print("   - unique_id: original unique identifier from CSV")
print("   - venue information: title, description, type, address, etc.")
print("   - city_id: FOREIGN KEY linking to cities table")
print("   - year: year of the venue (directly stored)")
print("   - No publications table needed for SC data")

# Step 4: Commit the structure and verify
print("\nStep 4: Saving database structure...")
conn.commit()

# Check that our tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("✓ Database structure saved")
print(f"✓ Tables created: {[table[0] for table in tables]}")

# Show table schemas
print("\n=== Database Schema Summary ===")
for table_name in ['cities', 'venues']:
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\n{table_name.upper()} TABLE:")
    for col in columns:
        print(f"   {col[1]} ({col[2]})")

# Close connection
conn.close()
print("\n=== Empty SC database structure ready! ===")
print("Next: Run import-data-sc.py to populate with SC data")

