
# Creating a new Flask application and importing the necessary libraries, 
# such as NumPy and Scikit-learn (if you are using a pre-trained model)

from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle


# Create an instance of the Flask class and define routes for the web page.
app = Flask(__name__)
model=pickle.load(open('stock-linreg-model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


# Creating another route that will handle the form submission and use the model to make a prediction

@app.route('/predict', methods=['POST'])
def predict():
    open = float(request.form['open'])
    high = float(request.form['high'])
    low = int(request.form['low'])
    volume = int(request.form['volume'])
    features = np.array([open,high,low,volume]).reshape(1, -1)
    probability=model.predict([[open,high,low,volume]])
    return render_template('predict.html', probability=probability)

# Finally, run the application using
if __name__ == '__main__':
    app.run(debug=True)
