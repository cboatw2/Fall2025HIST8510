# Database Reset Script
# History 8510 - Database Lesson Reset
# This script resets the database so all lesson scripts can be run again

import os
import sqlite3

print("=== Gay Guides Database Reset Tool ===")
print("This script will reset the database for fresh lesson execution")

# Check if database exists
db_file = 'gay_guides.db'
db_exists = os.path.exists(db_file)

if db_exists:
    print(f"\n✓ Found existing database: {db_file}")
    
    # Show current database contents before deletion
    print("\nCurrent database contents:")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check if tables exist and show counts
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if tables:
            print("   Tables found:")
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"   - {table_name}: {count} records")
        else:
            print("   - No tables found (empty database)")
            
        conn.close()
        
    except sqlite3.Error as e:
        print(f"   - Error reading database: {e}")
    
    # Delete the database file
    print(f"\nStep 1: Removing {db_file}...")
    try:
        os.remove(db_file)
        print("✓ Database file deleted successfully")
    except OSError as e:
        print(f"❌ Error deleting database: {e}")
        exit(1)
        
else:
    print(f"\n✓ No existing database found ({db_file})")

# Check for CSV data file
csv_file = 'co-rand-100.csv'
if os.path.exists(csv_file):
    print(f"✓ CSV data file found: {csv_file}")
else:
    print(f"⚠️  Warning: CSV data file not found: {csv_file}")
    print("   Make sure this file is present before running import-data.py")

# Verify scripts are present
scripts_to_check = [
    'create-db.py',
    'import-data.py', 
    'single-record.py'
]

print("\nChecking lesson scripts:")
all_scripts_present = True
for script in scripts_to_check:
    if os.path.exists(script):
        print(f"✓ {script}")
    else:
        print(f"❌ {script} - NOT FOUND")
        all_scripts_present = False

# Final status
print("\n=== Reset Complete ===")
print("Database has been reset!")
print("\nTo run the full lesson sequence:")
print("1. python3 create-db.py      # Creates empty database structure")
print("2. python3 single-record.py  # Adds a manual record")
print("3. python3 import-data.py    # Imports CSV data")


if not all_scripts_present:
    print("\n⚠️  Warning: Some lesson scripts are missing!")
    
if not os.path.exists(csv_file):
    print(f"\n⚠️  Warning: {csv_file} is required for import-data.py")

print("\n✓ Ready for fresh lesson execution!")
