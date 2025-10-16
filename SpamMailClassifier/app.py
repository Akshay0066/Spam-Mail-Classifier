from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['message']
    data = vectorizer.transform([msg])
    pred = model.predict(data)[0]
    return render_template('index.html', prediction_text=f"The message is: {pred.upper()}")

if __name__ == '__main__':
    app.run(debug=True)
