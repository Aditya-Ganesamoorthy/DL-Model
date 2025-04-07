from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# MongoDB Configuration
MONGO_URI = "mongodb+srv://vgugan16:gugan2004@cluster0.qyh1fuo.mongodb.net/dL?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "dL"
READINGS_COLLECTION = "Bookq"
MEAN_VALUES_COLLECTION = "mean_json"

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
readings_collection = db[READINGS_COLLECTION]
mean_values_collection = db[MEAN_VALUES_COLLECTION]

def run_eda(df):
    plot_paths = []
    
    if df.empty:
        return plot_paths

    # Create static directory if not exists
    static_dir = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Plot 1: Threshold vs Gas Emission Value
    plt.figure()
    sns.scatterplot(data=df, x="Gas Emission Value", y="Threshold", hue="Location")
    plot1_filename = 'plot1.png'
    plt.savefig(os.path.join(static_dir, plot1_filename))
    plt.close()
    plot_paths.append(plot1_filename)

    # Plot 2: Average Threshold per Location
    plt.figure()
    mean_by_location = df.groupby("Location")["Threshold"].mean().reset_index()
    sns.barplot(data=mean_by_location, x="Location", y="Threshold")
    plt.xticks(rotation=45)
    plot2_filename = 'plot2.png'
    plt.savefig(os.path.join(static_dir, plot2_filename))
    plt.close()
    plot_paths.append(plot2_filename)

    # Plot 3: Time Series of Daily Mean Values
    plt.figure()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date
    daily_mean = df.groupby('Date')[['Threshold', 'Gas Emission Value']].mean().reset_index()
    daily_mean['Date'] = pd.to_datetime(daily_mean['Date'])
    daily_mean.plot(x='Date', y=['Threshold', 'Gas Emission Value'], title="Daily Averages")
    plt.ylabel("Value")
    plt.grid(True)
    plot3_filename = 'plot3.png'
    plt.savefig(os.path.join(static_dir, plot3_filename))
    plt.close()
    plot_paths.append(plot3_filename)

    return plot_paths

def load_or_initialize_data():
    cursor = readings_collection.find({}, {'_id': 0})
    data = list(cursor)

    if data:
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date.astype(str)
    else:
        df = pd.DataFrame(columns=["Location", "Temperature", "Gas Emission Value", 
                                  "Threshold", "Analog", "Timestamp", "Date"])

    df['Location'] = df['Location'].fillna("Unknown")
    return df

def handle_location_encoding(df):
    label_encoder = LabelEncoder()
    if not df.empty and 'Location' in df.columns:
        unique_locations = df['Location'].unique().tolist()
        if "Unknown" not in unique_locations:
            unique_locations.append("Unknown")
        label_encoder.fit(unique_locations)
    return label_encoder

def train_model(df, label_encoder):
    if "Threshold" in df.columns and not df.empty and 'Location' in df.columns:
        try:
            df["Location_Encoded"] = label_encoder.transform(df["Location"])
            X = df[["Location_Encoded", "Temperature", "Gas Emission Value"]]
            y = df["Threshold"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            return model, mae, rmse
        except ValueError:
            return None, None, None
    return None, None, None

def save_mean_values():
    df = load_or_initialize_data()
    if not df.empty:
        mean_values = df.groupby('Date').agg({
            'Threshold': 'mean',
            'Gas Emission Value': 'mean'
        }).reset_index()

        mean_values_records = mean_values.to_dict(orient='records')

        for record in mean_values_records:
            mean_values_collection.update_one(
                {"Date": record["Date"]},
                {"$set": record},
                upsert=True
            )
from flask import jsonify

@app.route('/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        location = data.get('location', 'Unknown')
        temperature = float(data.get('temperature', 0))
        gas_emission = float(data.get('gas_emission', 0))

        df = load_or_initialize_data()
        label_encoder = handle_location_encoding(df)
        model, mae, rmse = train_model(df, label_encoder)

        try:
            location_encoded = label_encoder.transform([location])[0]
        except ValueError:
            new_classes = list(label_encoder.classes_) + [location]
            label_encoder.fit(new_classes)
            location_encoded = label_encoder.transform([location])[0]

        user_input = pd.DataFrame([{
            "Location_Encoded": location_encoded,
            "Temperature": temperature,
            "Gas Emission Value": gas_emission
        }])
        predicted_threshold = model.predict(user_input)[0]
        analog_value = 1 if gas_emission > predicted_threshold else 0

        return jsonify({
            "predicted_threshold": round(float(predicted_threshold), 2),
            "analog": analog_value
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/', methods=['GET', 'POST'])
def index():
    df = load_or_initialize_data()
    plot_paths = run_eda(df)
    label_encoder = handle_location_encoding(df)
    model, mae, rmse = train_model(df, label_encoder)

    predicted_threshold = None
    analog_value = None

    if request.method == 'POST':
        try:
            location = request.form.get('location', 'Unknown')
            temperature = float(request.form.get('temperature', 0))
            gas_emission = float(request.form.get('gas_emission', 0))

            try:
                location_encoded = label_encoder.transform([location])[0]
            except ValueError:
                new_classes = list(label_encoder.classes_) + [location]
                label_encoder.fit(new_classes)
                location_encoded = label_encoder.transform([location])[0]

            if model:
                user_input = pd.DataFrame([{
                    "Location_Encoded": location_encoded,
                    "Temperature": temperature,
                    "Gas Emission Value": gas_emission
                }])
                predicted_threshold = model.predict(user_input)[0]
            else:
                predicted_threshold = gas_emission * 1.1

            analog_value = 1 if gas_emission > predicted_threshold else 0

            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_entry = {
                "Location": location,
                "Temperature": temperature,
                "Gas Emission Value": gas_emission,
                "Threshold": float(predicted_threshold),
                "Analog": int(analog_value),
                "Timestamp": current_timestamp,
                "Date": datetime.now().date().isoformat()
            }

            readings_collection.insert_one(new_entry)
            save_mean_values()
            flash('Prediction saved successfully!', 'success')

        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

        return render_template('index.html',
                               plot_paths=plot_paths,
                               mae=mae,
                               rmse=rmse,
                               model_trained=model is not None,
                               predicted_threshold=round(predicted_threshold, 2),
                               analog_value=analog_value,
                               location=location,
                               temperature=temperature,
                               gas_emission=gas_emission)

    return render_template('index.html',
                           plot_paths=plot_paths,
                           mae=mae,
                           rmse=rmse,
                           model_trained=model is not None,
                           predicted_threshold=None,
                           analog_value=None)

if __name__ == '__main__':
    app.run(debug=True)
