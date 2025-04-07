import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
    if df.empty:
        st.warning("No data available for EDA.")
        return

    st.subheader("EDA: Emission Threshold Insights")

    st.write("### Threshold vs Gas Emission Value")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Gas Emission Value", y="Threshold", hue="Location", ax=ax1)
    st.pyplot(fig1)

    st.write("### Average Threshold per Location")
    mean_by_location = df.groupby("Location")["Threshold"].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(data=mean_by_location, x="Location", y="Threshold", ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.write("### Time Series of Daily Mean Values")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    daily_mean = df.groupby('Date')[['Threshold', 'Gas Emission Value']].mean().reset_index()
    daily_mean['Date'] = pd.to_datetime(daily_mean['Date'])

    fig3, ax3 = plt.subplots()
    daily_mean.plot(x='Date', y=['Threshold', 'Gas Emission Value'], ax=ax3, title="Daily Averages")
    ax3.set_ylabel("Value")
    ax3.grid(True)
    st.pyplot(fig3)

def load_or_initialize_data():
    cursor = readings_collection.find({}, {'_id': 0})
    data = list(cursor)

    if data:
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date.astype(str)
    else:
        df = pd.DataFrame(columns=["Location", "Temperature", "Gas Emission Value", "Threshold", "Analog", "Timestamp", "Date"])

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
            st.success(f"Model Metrics:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}")

            return model

        except ValueError as e:
            st.error(f"Model training error: {e}")
            return None

    st.warning("Insufficient data for model training")
    return None

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

        st.success("Daily mean values saved to MongoDB successfully!")

# Streamlit UI
df = load_or_initialize_data()
run_eda(df)
label_encoder = handle_location_encoding(df)
model = train_model(df, label_encoder)
st.title("Gas Emission Threshold Prediction System")

st.markdown("Enter data to predict threshold and store results.")

location = st.text_input("Location", value="Unknown")
temperature = st.number_input("Temperature", format="%.2f")
gas_emission = st.number_input("Gas Emission Value", format="%.2f")

if st.button("Predict and Save"):
    try:
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

        analog = 1 if gas_emission > predicted_threshold else 0

        st.success(f"Prediction Results:\nThreshold: {predicted_threshold:.2f}\nAnalog Value: {analog}")

        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entry = {
            "Location": location,
            "Temperature": temperature,
            "Gas Emission Value": gas_emission,
            "Threshold": float(predicted_threshold),
            "Analog": int(analog),
            "Timestamp": current_timestamp,
            "Date": datetime.now().date().isoformat()
        }

        readings_collection.insert_one(new_entry)
        st.success("Data saved to MongoDB successfully!")

        save_mean_values()

    except Exception as e:
        st.error(f"Error: {e}")