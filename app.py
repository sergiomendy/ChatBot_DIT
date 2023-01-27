from flask import Flask, render_template, request
import random
from chatbot import Chatbot
from keras.models import load_model
import numpy as np

p = Chatbot("intents.json")
model = load_model("ditbot.h5")

app = Flask(__name__)

#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    input = p.input_preprocess(userText)
    pred = np.argmax(model.predict(input))
    decoded_data = p.labelize().inverse_transform([pred])
    tag = decoded_data[0]
    random.choice(p.reponses.get(tag))
    return str(random.choice(p.reponses.get(tag)))

if __name__ == "__main__":
    app.run(debug=True)