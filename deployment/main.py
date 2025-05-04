from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load(r'C:\Users\himan\OneDrive\Desktop\Taxi-Fare-Prediciton\models\random_forest_model.pkl')  


scaler = StandardScaler()

def preprocess_input(passenger_count, trip_distance, hour, day, month, day_of_week, is_weekend):
    distance_km = trip_distance * 1.60934  
    
    
    input_data = pd.DataFrame({
        'passenger_count': [passenger_count],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'distance_km': [distance_km]
    })
   
    input_data_scaled = scaler.fit_transform(input_data)  
    return input_data_scaled

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        passenger_count = int(request.form["passenger_count"])
        trip_distance = float(request.form["trip_distance"])
        hour = int(request.form["hour"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        day_of_week = int(request.form["day_of_week"])
        is_weekend = 1 if day_of_week >= 5 else 0  

        
        input_data = preprocess_input(passenger_count, trip_distance, hour, day, month, day_of_week, is_weekend)
        prediction = model.predict(input_data)

       
        return render_template("index.html", prediction=f"Predicted Fare: ${prediction[0]:.2f}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)