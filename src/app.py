from flask import Flask, jsonify
from flask import request
from model import *

app = Flask(__name__)
class SingletonSentimentAnalysisModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonSentimentAnalysisModel, cls).__new__(cls, *args, **kwargs)
            cls._instance.model = SentimentAnalysisModel()
        return cls._instance

model = SingletonSentimentAnalysisModel().model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    result = model.predict(text)
    return jsonify({
                    "received_text": text,
                    "result": result
                    })

if __name__ == '__main__':
    app.run(debug=False)