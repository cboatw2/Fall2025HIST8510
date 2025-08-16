# Script 1: Creating Database Structure
# History 8510 - August 25
# This script creates the database tables but doesn't add any data yet

import sqlite3

print("=== Creating Gay Guides Database Structure ===")

# Step 1: Connect to database (creates file if it doesn't exist)
print("\nStep 1: Connecting to database...")
conn = sqlite3.connect('gay_guides.db')
cursor = conn.cursor()
print("✓ Connected to gay_guides.db")

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

# Step 3: Create publications table
print("\nStep 3: Creating publications table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS publications (
    pub_id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_name TEXT NOT NULL,
    year INTEGER NOT NULL,
    UNIQUE(publication_name, year)
)
''')
print("✓ Created publications table")
print("   - pub_id: unique identifier (auto-generated)")
print("   - publication_name: name of the guidebook")
print("   - year: year of publication")
print("   - UNIQUE constraint: prevents duplicate publication/year combinations")

# Step 4: Create venues table with foreign keys
print("\nStep 4: Creating venues table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS venues (
    venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    type TEXT,
    address TEXT,
    city_id INTEGER,
    pub_id INTEGER,
    stars TEXT,
    star_type TEXT,
    amenity_features TEXT,
    status TEXT,
    geo_address TEXT,
    lat REAL,
    lon REAL,
    FOREIGN KEY (city_id) REFERENCES cities (city_id),
    FOREIGN KEY (pub_id) REFERENCES publications (pub_id)
)
''')
print("✓ Created venues table")
print("   - venue_id: unique identifier (auto-generated)")
print("   - venue information: title, description, type, address, etc.")
print("   - city_id: FOREIGN KEY linking to cities table")
print("   - pub_id: FOREIGN KEY linking to publications table")
print("   - FOREIGN KEYs ensure data integrity")

# Step 5: Commit the structure and verify
print("\nStep 5: Saving database structure...")
conn.commit()

# Check that our tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("✓ Database structure saved")
print(f"✓ Tables created: {[table[0] for table in tables]}")

# Show table schemas
print("\n=== Database Schema Summary ===")
for table_name in ['cities', 'publications', 'venues']:
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\n{table_name.upper()} TABLE:")
    for col in columns:
        print(f"   {col[1]} ({col[2]})")

# Close connection
conn.close()
print("\n=== Empty database structure ready! ===")
print("Next: Open gay_guides.db in VS Code SQLite viewer to explore the structure")