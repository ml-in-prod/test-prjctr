# Text readability assessment

This is a test task for 'Machine Learning in production' course on projector
The repo consists of 2 main parts: model.py and server.py

* model.py is responsible for training a simple autokeras text regression model (we choose regression over classification because we need to work with continuous values)
* server.py uses that trained model to host a simple http api that allows user to determine readability of arbitrary piece of text

In order to reproduce the results you need to run through following steps:

```sh
# Install autokeras to work with ML model
pip install autokeras
```

```sh
# Install fastapi and uvicorn to work with the server
pip install fastapi "uvicorn[standard]"
```

```sh
# Train the model
python model.py
```

```sh
# Run the server
uvicorn server:app --reload
```

Functionality will be available on /test endpoint where you should include your text into the 'text' field in the JSON body of the POST request