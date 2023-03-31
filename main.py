from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

app = FastAPI()

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(model, "model.joblib")

# Define input parameters for the API endpoint
class InputParams(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the API endpoint for predicting iris species
@app.post("/predict")
async def predict(input_params: InputParams):
    # Load the trained model from disk
    model = joblib.load("model.joblib")
    # Convert input parameters to a numpy array
    input_array = [[
        input_params.sepal_length, input_params.sepal_width,
        input_params.petal_length, input_params.petal_width
    ]]
    # Use the model to predict the iris species
    prediction = model.predict(input_array)
    # Map predicted class index to class name
    target_names = iris.target_names.tolist()
    predicted_class = target_names[prediction[0]]
    # Return the predicted class as a JSON response
    return {"predicted_class": predicted_class}
