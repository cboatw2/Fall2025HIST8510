"""
Simple Flask Web Application Demo
================================

How to run:
1. Install Flask: pip install -r requirements.txt
2. Run the app: python3 app.py
3. Open browser: http://localhost:5001

Routes:
- / (home page): Shows a list of 10 venues
- /venue/<id> (venue detail): Shows detailed info for one venue

Database:
- Uses SQLite database: ../Week3-SQL-Queries/sc_gay_guides.db
- Tables: venues, cities
- Joins venues with cities to get location information
"""

# Import Flask and other necessary modules
from flask import Flask, render_template  # Flask for web framework, render_template for HTML templates
import sqlite3  # For database operations
import os  # For file system operations (not used in this simple example)

# Create a Flask application instance
# __name__ tells Flask where to find templates and static files
app = Flask(__name__)

# Database configuration
# This is the path to our SQLite database file
# We're using a relative path to go up one directory to Week3-SQL-Queries
DB_PATH = 'sc_gay_guides.db'

def get_db_connection():
    """
    Create a connection to the SQLite database
    
    This function:
    1. Opens a connection to the database file
    2. Sets row_factory to sqlite3.Row so we can access columns by name
       (instead of just by index number)
    3. Returns the connection object
    
    Returns:
        sqlite3.Connection: A database connection object
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name like row['title']
    return conn

@app.route('/')
def index():
    """
    Home page route - displays a list of venues
    
    This is the main page of our web application. When someone visits
    the root URL (http://localhost:5001/), this function runs.
    
    What this function does:
    1. Connects to the database
    2. Runs a SQL query to get venue information
    3. Closes the database connection
    4. Renders an HTML template with the data
    
    Returns:
        HTML page: The rendered index.html template with venue data
    """
    # Get a connection to our database
    conn = get_db_connection()
    
    # Execute a SQL query to get venue data
    # This query:
    # - SELECTs specific columns from venues (v) and cities (c) tables
    # - JOINs the venues and cities tables on city_id
    # - FILTERs out venues with empty or null titles
    # - LIMITs results to 10 venues
    venues = conn.execute('''
        SELECT v.title, v.description, v.type, v.street_address, 
               c.city_name, c.state, v.year
        FROM venues v 
        JOIN cities c ON v.city_id = c.city_id 
        WHERE v.title IS NOT NULL AND v.title != ''
        LIMIT 10
    ''').fetchall()  # fetchall() gets all the results as a list of Row objects
    
    # Always close database connections when done
    conn.close()
    
    # Render the HTML template and pass the venues data to it
    # The template can access the 'venues' variable
    return render_template('index.html', venues=venues)

@app.route('/venue/<int:venue_id>')
def venue_detail(venue_id):
    """
    Venue detail page route - displays detailed information for one venue
    
    This route handles URLs like /venue/123 where 123 is the venue ID.
    Flask automatically extracts the venue_id from the URL and passes it
    as a parameter to this function.
    
    Parameters:
        venue_id (int): The ID of the venue to display (from the URL)
    
    What this function does:
    1. Connects to the database
    2. Runs a SQL query to get ONE specific venue by its ID
    3. Closes the database connection
    4. Checks if the venue was found
    5. Either renders the detail template or returns a 404 error
    
    Returns:
        HTML page: Either the venue detail page or a 404 error message
    """
    # Get a connection to our database
    conn = get_db_connection()
    
    # Execute a SQL query to get ONE specific venue
    # This query:
    # - SELECTs ALL columns from venues (v.*) plus city info
    # - JOINs venues and cities tables
    # - WHERE clause filters by the specific venue_id
    # - Uses ? placeholder to prevent SQL injection attacks
    # - fetchone() gets just one result (or None if not found)
    venue = conn.execute('''
        SELECT v.*, c.city_name, c.state
        FROM venues v 
        JOIN cities c ON v.city_id = c.city_id 
        WHERE v.venue_id = ?
    ''', (venue_id,)).fetchone()  # The (venue_id,) is a tuple with the parameter
    
    # Always close database connections when done
    conn.close()
    
    # Check if we found a venue with that ID
    if venue is None:
        # Return a 404 error if venue doesn't exist
        return "Venue not found", 404
    
    # Render the detail template and pass the venue data to it
    return render_template('venue_detail.html', venue=venue)

# New route to display all unique cities
@app.route('/cities')
def cities():
    """
    Cities page route - displays a list of unique cities
    """
    conn = get_db_connection()
    cities = conn.execute('SELECT DISTINCT city_name FROM cities ORDER BY city_name').fetchall()
    conn.close()
    return render_template('cities.html', cities=cities)

# This block only runs when the script is executed directly
# (not when imported as a module)
if __name__ == '__main__':
    # Start the Flask development server
    # Parameters explained:
    # - debug=True: Enables debug mode (auto-reloads on changes, shows detailed errors)
    # - host='0.0.0.0': Makes the server accessible from any IP address
    # - port=5001: Runs on port 5001 (changed from 5000 to avoid conflicts)
    app.run(debug=True, host='0.0.0.0', port=5001)
