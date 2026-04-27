import pickle
import inspect

with open("random_forest_model_amp.pkl", "rb") as f:
    model = pickle.load(f)

# Get the type of the model
print("Model type:", type(model))

# List the attributes and methods of the model
print("Model attributes and methods:", dir(model))

# Optionally, inspect the model's methods
for method_name in dir(model):
    if callable(getattr(model, method_name)):
        print(f"Method: {method_name}")
        print(inspect.signature(getattr(model, method_name)))