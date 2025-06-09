# Import libraries
import requests  #This library helps us to fetch data from APIs
import pandas as pd  #For handling and analysing data
import numpy as np  #For numerical operation
from sklearn.model_selection import train_test_split  #To split data into training and testing set
from sklearn.preprocessing import LabelEncoder  #To convert categerical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  #Models for classification and regression tasks
from sklearn.metrics import mean_squared_error #To measure the accuracy of our predictions
from datetime import datetime, timedelta  #to handle date and time
import pytz
import matplotlib.pyplot as plt
import seaborn as sns


API_KEY = 'YOUR API KEY'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'  #Base url for making api request


def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metrics"
    response = requests.get(url)  #send the get request to API
    data = response.json()
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'Wind_Gust_Speed': data['wind']['speed']
    }


# Read historical Data
def read_historical_data(filename):
    df = pd.read_csv(filename) #load csv file into dataFrame
    df = df.dropna () #remove rows wit missing values
    df = df.drop_duplicates()
    return df


# Prepare data for training
def prepare_data(data):
    le= LabelEncoder() #create a LabelEncoder instance
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    
    #define the feature variable and target variablesst
    x = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] #feature variables
    y = data['RainTomorrow'] #target variable
    return x, y, le #return feture variable, target variable and the label encoder


# Train rain prediction model
def train_rain_model(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train) #train the model
    y_pred = model.predict(X_test) #to make predictions on test set
    print("Mean Squared Error for Rain Model")
    print (mean_squared_error(y_test, y_pred))
    return model


# Prepare regession data
def prepare_regression_data(data, feature):
    x, y = [], [] #initialize list for feature and target values
    for i in range(len(data) - 1):
        x.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
        
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
     # Take last 'lookback' actual values for comparison
    expected_values = data[feature].iloc[-lookback:].tolist()
    return x, y, expected_values


# Train regession model
def train_regression_model(x, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x, y)
    return model


# Predict future
def predict_future (model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions [1:]


#Graph plot
def plot_future_weather(future_times, future_temp, future_humidity, city):
    plt.figure(figsize=(12, 5))
    
    # Temperature Plot
    plt.subplot(1, 2, 1)
    sns.lineplot(x=future_times, y=future_temp, marker='o', color='tomato')
    plt.title(f'Future Temperature Forecast - {city}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Humidity Plot
    plt.subplot(1, 2, 2)
    sns.lineplot(x=future_times, y=future_humidity, marker='o', color='skyblue')
    plt.title(f'Future Humidity Forecast - {city}')
    plt.xlabel('Time')
    plt.ylabel('Humidity (%)')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(time_labels, predicted, actual, ylabel, title, city):
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=time_labels, y=actual, marker='o', label='Expected', color='green')
    sns.lineplot(x=time_labels, y=predicted, marker='o', label='Predicted', color='orange')
    
    plt.title(f"{title} - Predicted vs Expected ({city})", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



# Weather analysis function
def weather_view():
    city = input('Enter any city name: ')
    current_weather = get_current_weather(city)
    
    #load historical data
    historical_data = read_historical_data('weather.csv')
    
    #prepae and train the rain prediction model
    x, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(x, y)

    #map wind direction to campass points
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

    compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather ['Wind_Gust_Speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp'],
    }

    current_df = pd.DataFrame([current_data])

    #rain prediction
    rain_prediction = rain_model.predict(current_df)[0]
    
    #prepare regression model for temperature and humidity
    x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)

    #predict future temperature and humidity
    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future (hum_model, current_weather['humidity'])
    
    #prepare time for future predictions
    timezone = pytz.timezone ('Asia/Kolkata')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta (hours=i)).strftime("%H:00") for i in range(5)]
    
    #Display results
    print(f"City: {city}, {current_weather['country']}")
    print (f"Current Temperature: {(current_weather['current_temp'])/10}°C")
    print (f"Feels Like: {(current_weather['feels_like'])/10}°C")
    print (f"Minimum Temperature: {(current_weather['temp_min'])/10}°C")
    print (f"Maximum Temperature: {(current_weather['temp_max'])/10}°C")
    print (f"Humidity: {current_weather['humidity']}%")
    print (f"Weather Prediction: {current_weather['description']}")
    print(f"Rain Prediction: {'Yes' if rain_prediction else 'No'}")
    
    print("\nFuture Temperature Predictions: ")


    for time, temp in zip(future_times, future_temp):
        print(f"{time}: {round (temp, 1)}°C")
        
    print("\nFuture Humidity Predictions: ")
    
    for time, humidity in zip(future_times, future_humidity):
        print(f" {time}: {round (humidity, 1)}%")

    # Plot graphs
    plot_future_weather(future_times, future_temp, future_humidity, city)

    # Comparison Graphs
    plot_prediction_vs_actual(future_times, future_temp, list(historical_data['Temp'].tail(5)), "Temperature (°C)", "Temperature", city)
    plot_prediction_vs_actual(future_times, future_humidity, list(historical_data['Humidity'].tail(5)), "Humidity (%)", "Humidity", city)


weather_view()
