import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('winequality-red.csv')

# Prepare the features and target
X = df.drop('quality', axis=1)  # Features
y = df['quality']  # Target variable

# Train a RandomForestClassifier model (if not done yet)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save the model and scaler using pickle
with open('wine_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the model and scaler
with open('wine_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create the main window
root = tk.Tk()
root.title("Wine Quality Prediction")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")  # Full screen

# Validation function to allow only numeric input
def validate_numeric_input(action, value_if_allowed):
    if action == '1':  # '1' means insert action
        try:
            float(value_if_allowed)  # Attempt to convert to float
            return True
        except ValueError:
            return False
    else:
        return True  # Allow backspace and deletion

# Register the validation function
vcmd = (root.register(validate_numeric_input), '%d', '%P')

# Create the GUI application using Tkinter
def predict_quality():
    try:
        # Get user input
        fixed_acidity = float(entry_fixed_acidity.get())
        volatile_acidity = float(entry_volatile_acidity.get())
        citric_acid = float(entry_citric_acid.get())
        residual_sugar = float(entry_residual_sugar.get())
        chlorides = float(entry_chlorides.get())
        free_sulfur_dioxide = float(entry_free_sulfur_dioxide.get())
        total_sulfur_dioxide = float(entry_total_sulfur_dioxide.get())
        density = float(entry_density.get())
        pH = float(entry_pH.get())
        sulphates = float(entry_sulphates.get())
        alcohol = float(entry_alcohol.get())
        
        # Prepare the feature list
        features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
        
        # Scale the features
        scaled_features = scaler.transform([features])
        
        # Predict the wine quality
        prediction = model.predict(scaled_features)
        
        # Show the prediction result
        messagebox.showinfo("Prediction Result", f"Predicted Wine Quality: {prediction[0]}")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields!")

# Function to clear all entry fields
def clear_entries():
    for entry in entries:
        entry.delete(0, tk.END)  # Clear each entry field

# Open the background image for the whole window
background_image = Image.open(r"bg.jpeg")
background_image = background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)

# Set the background image for the main window
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Create labels and entry widgets for each wine attribute
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.8)  # Center the frame in the window

# Open the background image for the frame
frame_bg_image = Image.open(r"bg.jpeg")
frame_bg_image = frame_bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
frame_bg_photo = ImageTk.PhotoImage(frame_bg_image)

# Set the background image for the frame
frame_background_label = tk.Label(frame, image=frame_bg_photo)
frame_background_label.place(relwidth=1, relheight=1)

# Labels for each   
labels = ["Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", "Chlorides", 
          "Free Sulfur Dioxide", "Total Sulfur Dioxide", "Density", "pH", "Sulphates", "Alcohol"]
entries = []

# Create the labels and entries for the attributes
for i, label_text in enumerate(labels):
    label = tk.Label(frame, text=label_text, font=('Arial', 14), bg="white", fg="black")
    label.grid(row=i//2, column=2*(i%2), padx=10, pady=10, sticky='w')
    entry = tk.Entry(frame, font=('Arial', 14), bg="white", fg="black", bd=2, relief="solid",
                     validate="key", validatecommand=vcmd)  # Add validation to entry
    entry.grid(row=i//2, column=2*(i%2) + 1, padx=10, pady=10)
    entries.append(entry)

# Assign each entry to a variable for easier access
entry_fixed_acidity, entry_volatile_acidity, entry_citric_acid, entry_residual_sugar, entry_chlorides, \
entry_free_sulfur_dioxide, entry_total_sulfur_dioxide, entry_density, entry_pH, entry_sulphates, entry_alcohol = entries

# Function to move focus to next entry field
def on_enter(event, next_widget):
    next_widget.focus()

# Bind the "Enter" key to move focus to the next entry
entry_fixed_acidity.bind("<Return>", lambda event: on_enter(event, entry_volatile_acidity))
entry_volatile_acidity.bind("<Return>", lambda event: on_enter(event, entry_citric_acid))
entry_citric_acid.bind("<Return>", lambda event: on_enter(event, entry_residual_sugar))
entry_residual_sugar.bind("<Return>", lambda event: on_enter(event, entry_chlorides))
entry_chlorides.bind("<Return>", lambda event: on_enter(event, entry_free_sulfur_dioxide))
entry_free_sulfur_dioxide.bind("<Return>", lambda event: on_enter(event, entry_total_sulfur_dioxide))
entry_total_sulfur_dioxide.bind("<Return>", lambda event: on_enter(event, entry_density))
entry_density.bind("<Return>", lambda event: on_enter(event, entry_pH))
entry_pH.bind("<Return>", lambda event: on_enter(event, entry_sulphates))
entry_sulphates.bind("<Return>", lambda event: on_enter(event, entry_alcohol))

# Create a button to predict the wine quality
predict_button = tk.Button(root, text="Predict Quality", command=predict_quality, font=('Arial', 16, 'bold'), bg="#FFFFFF", fg="black", relief="raised", bd=5)
predict_button.place(relx=0.4, rely=0.7, anchor="center")  # Position the button in the lower part of the window

# Create a "Clear" button to reset the input fields
clear_button = tk.Button(root, text="Clear", command=clear_entries, font=('Arial', 16, 'bold'), bg="#FFFFFF", fg="black", relief="raised", bd=5)
clear_button.place(relx=0.6, rely=0.7, anchor="center")  # Position it next to the "Predict Quality" button

# Run the application
root.mainloop()
