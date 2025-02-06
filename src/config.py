class Config:
    """Configuration settings for the Flask application."""
    
    DEBUG = True  # Enable debug mode
    TESTING = False  # Disable testing mode
    SECRET_KEY = 'your_secret_key'  # Replace with a strong secret key
    DATABASE_URI = 'sqlite:///your_database.db'  # Database URI for SQLite
    # Add other configuration variables as needed