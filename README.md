# README.md

# Flask Simple API

This project is a simple Flask API that demonstrates how to create a basic web service with a single endpoint.

## Project Structure

```
machine-learning
├── src
│   ├── app.py
│   ├── config.py
│   ├── model.py
|   └── temp.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.x
- Flask

## Installation

1. Clone the repository:

```
   git clone <repository-url> cd machine-learning
```

2. Install the required packages:

```
   pip install -r requirements.txt
```

If you encounter an error regarding `werkzeug.urls` (e.g., "cannot import name 'url_quote'"), downgrade werkzeug with:

```
   pip install werkzeug==2.0.3
```

## Running the Application

To run the Flask application, execute:

```
   python src/app.py
```

The application will start on `http://127.0.0.1:5000/`.

## Usage

The API has a single endpoint:

- `POST /predict`: Receive some text and returns a sentiment from received text (Negative, Neutral, or Positive).

## License

This project is licensed under the MIT License.
