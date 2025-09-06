# Simple Flask App Demo

This is a very simple Flask application that demonstrates basic web development concepts using the SC Gay Guides database.

## What this app does

- Displays a list of venues from the `sc_gay_guides.db` database
- Shows venue details when you click on individual venues
- Demonstrates Flask routing, templates, and database connections

## How to run the app

1. Install Flask:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

## What you'll see

- A list of 10 venues from the database
- Each venue shows title, description, address, location, year, and type
- Clean, simple styling to make it easy to read

## Key Flask concepts demonstrated

- **Routes**: `/` (home page) and `/venue/<id>` (venue details)
- **Templates**: HTML templates with Jinja2 templating
- **Database connections**: SQLite3 integration
- **Template inheritance**: Base template with child templates
- **Error handling**: 404 error for non-existent venues

This is a great starting point for understanding how Flask works!
