from dotenv import load_dotenv
from flask import Flask

load_dotenv()
from flask_cors import CORS
from routes import register_routes

app = Flask(__name__)
CORS(app)

# Register routes
register_routes(app)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
