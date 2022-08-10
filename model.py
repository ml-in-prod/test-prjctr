import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
import tensorflow as tf

import autokeras as ak

train_dataset = pd.read_csv('./train.csv',
                            na_values="?", comment='\t',
                            sep=",", skipinitialspace=True)

test_dataset = pd.read_csv('./test.csv',
                           na_values="?", comment='\t',
                           sep=",", skipinitialspace=True)

x_test_dataset = test_dataset.excerpt


X = train_dataset.excerpt
y = train_dataset.target

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

x_train = np.array(train_X)
y_train = np.array(train_y)

x_test = np.array(val_X)
y_test = np.array(val_y)

# Initialize the text regressor.
reg = ak.TextRegressor(overwrite=True, max_trials=5)
# Feed the text regressor with training data.
reg.fit(x_train, y_train, epochs=5)
# Predict with the best model.
predicted_y = reg.predict(x_test)
# Evaluate the best model with testing data.
print(reg.evaluate(x_test, y_test))

print(mean_absolute_error(predicted_y, y_test))

model = reg.export_model()

print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")


loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(x_test_dataset)
print(predicted_y)
