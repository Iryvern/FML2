from flask import Flask, jsonify
from flask_cors import CORS
from imports import *

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/api/message', methods=['GET'])
def get_message():
    return jsonify({"message": "Hello from Flask!"})

if __name__ == '__main__':
    app.run(debug=True)
