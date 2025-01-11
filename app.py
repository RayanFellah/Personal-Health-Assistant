from flask import Flask, render_template, request, jsonify
import chatbot
# Initialize Flask app
app = Flask("ChatBot")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json["message"]
    intents_list = chatbot.predict_class(user_input)
    response = chatbot.get_response(intents_list, chatbot.intents)
    return jsonify({"response": response})


if __name__ == "__main__":
    import os
    print(os.listdir("templates"))
    app.run(debug=True)