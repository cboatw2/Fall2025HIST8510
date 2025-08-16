
import sqlite3

print("=== Adding One Record Manually ===")

# Connect to database
conn = sqlite3.connect('gay_guides.db')
cursor = conn.cursor()

# Step 1: Add a city
print("Step 1: Adding San Francisco, CA...")
cursor.execute('''
INSERT INTO cities (city_name, state)
VALUES (?, ?)
''', ("San Francisco", "CA"))
city_id = cursor.lastrowid
print(f"✓ Added with city_id: {city_id}")

# Step 2: Add a publication
print("\nStep 2: Adding Damron Address Book 1965...")
cursor.execute('''
INSERT INTO publications (publication_name, year)
VALUES (?, ?)
''', ("Damron Address Book", 1965))
pub_id = cursor.lastrowid
print(f"✓ Added with pub_id: {pub_id}")

# Step 3: Add a venue using the foreign keys
print("\nStep 3: Adding The Black Cat venue...")
cursor.execute('''
INSERT INTO venues (
    record_id, title, description, type, address,
    city_id, pub_id, stars, amenity_features, lat, lon
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', (
    "SF001",
    "The Black Cat",
    "Famous bohemian bar",
    "Bar",
    "710 Montgomery Street",
    city_id,      # Links to San Francisco
    pub_id,       # Links to Damron 1965
    "***",
    "Dancing, Live Music",
    37.7962,
    -122.4035
))
print("✓ Added venue with connections to city and publication")

# Save and show the connected data
conn.commit()