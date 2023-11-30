import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, Label, Button
from tkinter import filedialog

# Function to scrape data from Wikipedia
def scrape_wikipedia_data():
    start_url = "https://en.wikipedia.org/wiki/List_of_brightest_stars_and_other_record_stars"
    page = requests.get(start_url)
    soup = BeautifulSoup(page.text, 'html.parser')

    headers = ["Proper name", "Distance", "Mass", "Radius", "Luminosity"]
    star_data = []

    star_table = soup.find('table')
    table_rows = star_table.find_all('tr')[1:]  # Skip the header row

    for tr in table_rows:
        row = [i.text.rstrip() for i in tr.find_all('td')]
        star_data.append(row)

    star_names = [data[1] for data in star_data]
    distances = [float(data[3].replace(',', '').replace('—', '0')) for data in star_data]
    masses = [float(data[5].replace(',', '').replace('—', '0')) for data in star_data]
    radii = [float(data[6].replace(',', '').replace('—', '0')) for data in star_data]
    luminosities = [float(data[7].replace(',', '').replace('—', '0')) for data in star_data]

    df = pd.DataFrame(list(zip(star_names, distances, masses, radii, luminosities)),
                      columns=['Star_name', 'Distance', 'Mass', 'Radius', 'Luminosity'])

    return df

# Function to perform regression using TensorFlow and Keras
def perform_regression(data):
    X = data[['Distance', 'Mass', 'Radius']]
    y = data['Luminosity']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simple neural network for regression using Keras
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model on the test set
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Visualize the predictions vs. actual values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions.flatten())
    plt.title('Actual vs. Predicted Luminosity')
    plt.xlabel('Actual Luminosity')
    plt.ylabel('Predicted Luminosity')
    plt.show()

# Function to create a simple GUI for data loading
def create_gui():
    root = Tk()
    root.title("Star Data GUI")

    def load_data():
        file_path = filedialog.askopenfilename()
        if file_path:
            data = pd.read_csv(file_path)
            print(data.head())

    label = Label(root, text="Star Data Analysis")
    label.pack()

    load_button = Button(root, text="Load Data", command=load_data)
    load_button.pack()

    root.mainloop()

def main():
    # Scrape data from Wikipedia
    star_data = scrape_wikipedia_data()

    # Save the scraped data to a CSV file
    star_data.to_csv('bright_stars_advanced.csv', index=False)

    # Perform regression using TensorFlow and Keras
    perform_regression(star_data)

    # Create a simple GUI for data loading
    create_gui()

if __name__ == "__main__":
    main()
