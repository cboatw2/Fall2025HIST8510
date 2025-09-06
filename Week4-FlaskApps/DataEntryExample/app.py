"""
Flask Data Entry Application Demo
================================

This Flask application demonstrates how to create web forms that allow users
to add new data to a database. It shows students how to:

Key Concepts Demonstrated:
- GET requests: Displaying forms
- POST requests: Handling form submissions
- Form validation: Checking user input
- Database INSERT operations: Adding new records
- Flash messages: Providing user feedback
- Redirects: Sending users to different pages after actions

Routes:
- / (home page): Shows list of venues and link to add new venue
- /add_venue (GET): Displays the form to add a new venue
- /add_venue (POST): Processes the form data and adds venue to database
- /venue/<id>: Shows details for a specific venue

Database Operations:
- SELECT: Reading existing venues
- INSERT: Adding new venues and cities
- Foreign key relationships: Linking venues to cities
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import os

# Create Flask application instance
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Database configuration
DB_PATH = 'sc_gay_guides.db'  # Database in same directory as app

def get_db_connection():
    """
    Create a connection to the SQLite database
    
    Returns:
        sqlite3.Connection: A database connection object
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """
    Home page - displays list of venues with option to add new ones
    
    Returns:
        HTML page: List of venues with "Add New Venue" link
    """
    conn = get_db_connection()
    
    # Get all venues with their city information
    venues = conn.execute('''
        SELECT v.venue_id, v.title, v.description, v.type, v.street_address, 
               c.city_name, c.state, v.year
        FROM venues v 
        JOIN cities c ON v.city_id = c.city_id 
        WHERE v.title IS NOT NULL AND v.title != ''
        ORDER BY v.title
        LIMIT 20
    ''').fetchall()
    
    conn.close()
    
    return render_template('index.html', venues=venues)

@app.route('/add_venue', methods=['GET', 'POST'])
def add_venue():
    """
    Add new venue route - handles both displaying form and processing submission
    
    GET: Shows the form to add a new venue
    POST: Processes the form data and adds venue to database
    
    Returns:
        HTML page: Either the form (GET) or redirect to home page (POST)
    """
    if request.method == 'POST':
        # Handle form submission
        return handle_venue_submission()
    else:
        # Display the form
        return show_venue_form()

def show_venue_form():
    """
    Display the form for adding a new venue
    
    Returns:
        HTML page: The venue form
    """
    conn = get_db_connection()
    
    # Get all cities for the dropdown
    cities = conn.execute('SELECT city_id, city_name, state FROM cities ORDER BY city_name').fetchall()
    
    conn.close()
    
    return render_template('add_venue.html', cities=cities)

def handle_venue_submission():
    """
    Process the form data and add new venue to database
    
    Returns:
        Redirect: To home page with success/error message
    """
    # Get form data
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    venue_type = request.form.get('type', '').strip()
    street_address = request.form.get('street_address', '').strip()
    city_id = request.form.get('city_id', '').strip()
    year = request.form.get('year', '').strip()
    
    # Basic validation
    if not title:
        flash('Title is required!', 'error')
        return redirect(url_for('add_venue'))
    
    if not city_id:
        flash('City is required!', 'error')
        return redirect(url_for('add_venue'))
    
    try:
        conn = get_db_connection()
        
        # Insert the new venue
        conn.execute('''
            INSERT INTO venues (title, description, type, street_address, city_id, year)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, description, venue_type, street_address, city_id, year))
        
        # Commit the changes
        conn.commit()
        conn.close()
        
        # Success message
        flash(f'Successfully added venue: {title}', 'success')
        
    except Exception as e:
        flash(f'Error adding venue: {str(e)}', 'error')
    
    # Redirect back to home page
    return redirect(url_for('index'))

@app.route('/venue/<int:venue_id>')
def venue_detail(venue_id):
    """
    Display details for a specific venue
    
    Parameters:
        venue_id (int): The ID of the venue to display
    
    Returns:
        HTML page: Venue details or 404 error
    """
    conn = get_db_connection()
    
    venue = conn.execute('''
        SELECT v.*, c.city_name, c.state
        FROM venues v 
        JOIN cities c ON v.city_id = c.city_id 
        WHERE v.venue_id = ?
    ''', (venue_id,)).fetchone()
    
    conn.close()
    
    if venue is None:
        flash('Venue not found!', 'error')
        return redirect(url_for('index'))
    
    return render_template('venue_detail.html', venue=venue)

if __name__ == '__main__':
    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5002)
