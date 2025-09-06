# Flask Data Entry Application Demo

This Flask application demonstrates how to create web forms that allow users to add new data to a database. It's designed to show students the fundamentals of data entry in web applications.

## What this app demonstrates

- **GET requests**: Displaying forms to users
- **POST requests**: Processing form submissions
- **Form validation**: Checking user input before saving
- **Database INSERT operations**: Adding new records to the database
- **Flash messages**: Providing user feedback
- **Redirects**: Sending users to different pages after actions
- **Foreign key relationships**: Linking venues to cities

## How to run the app

1. Install Flask:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python3 app.py
   ```

3. Open your web browser and go to:
   ```
   http://localhost:5002
   ```

## What you can do

### View Venues
- See a list of all venues in the database
- Click "View Details" to see full information for any venue

### Add New Venues
- Click "Add New Venue" button
- Fill out the form with venue information:
  - **Title** (required): Name of the venue
  - **Description**: Details about the venue
  - **Type**: Category (Nightclub, Bar, Restaurant, etc.)
  - **Street Address**: Physical address
  - **City** (required): Select from existing cities
  - **Year**: When the venue was established
- Submit the form to add the venue to the database

## Key Flask concepts demonstrated

### Routes with Multiple Methods
```python
@app.route('/add_venue', methods=['GET', 'POST'])
def add_venue():
    if request.method == 'POST':
        # Handle form submission
    else:
        # Display the form
```

### Form Data Handling
```python
title = request.form.get('title', '').strip()
city_id = request.form.get('city_id', '').strip()
```

### Database INSERT Operations
```python
conn.execute('''
    INSERT INTO venues (title, description, type, street_address, city_id, year)
    VALUES (?, ?, ?, ?, ?, ?)
''', (title, description, venue_type, street_address, city_id, year))
```

### User Feedback
```python
flash(f'Successfully added venue: {title}', 'success')
```

### Redirects
```python
return redirect(url_for('index'))
```

## Database Structure

The app works with two main tables:
- **venues**: Contains venue information (title, description, type, address, etc.)
- **cities**: Contains city information (city_name, state)

Venues are linked to cities through a foreign key relationship (`city_id`).

## Educational Value

This example shows students:
1. How web forms work in Flask
2. The difference between GET and POST requests
3. How to validate user input
4. How to insert data into databases
5. How to provide user feedback
6. How to handle errors gracefully
7. How to use redirects to improve user experience

Perfect for understanding the basics of data entry in web applications!
