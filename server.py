import json
from typing import Union
import autokeras as ak
from fastapi import FastAPI
from pydantic import BaseModel

from tensorflow.keras.models import load_model


class Item(BaseModel):
    text: str


app = FastAPI()
loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)


@app.post("/test/")
async def create_item(item: Item):
    prediction = loaded_model.predict([item.text])

    return json.dumps(prediction.tolist())
