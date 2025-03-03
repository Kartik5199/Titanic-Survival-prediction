from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Home route with form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    pclass = int(request.form['pclass'])
    sex = 1 if request.form['sex'].lower() == "male" else 0
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked'].upper()

    # Encode 'Embarked' values (Assuming 'S' is default)
    if embarked == "C":
        embarked = 0
    elif embarked == "Q":
        embarked = 1
    else:  # 'S'
        embarked = 2

    # Create the input array (NO SCALING)
    user_input = np.array([pclass,sex,age,sibsp,parch,fare,embarked]).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(user_input)
    result = "Survived" if prediction[0] == 1 else "Not Survived"

    return render_template('result.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
