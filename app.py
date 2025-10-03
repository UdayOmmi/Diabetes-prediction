from flask import Flask, render_template, request, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
            "BMI","DiabetesPedigreeFunction","Age"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        try:
            
            user_input = [float(request.form[feature]) for feature in features]
            user_input = np.array(user_input).reshape(1, -1)

            user_input_scaled = scaler.transform(user_input)

          
            prediction = model.predict(user_input_scaled)
            if prediction[0] == 0:
                result = "The person is NOT diabetic."
            else:
                result = "The person IS diabetic."

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, features=features)

if __name__ == "__main__":
    app.run(debug=True)
