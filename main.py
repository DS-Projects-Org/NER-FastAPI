# Copyright © 2020 Abubakar Yagoub

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# a simple NER API built using fastAPI and deployed to heroku

import pickle as pk

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()

# load saved model
model = tf.keras.models.load_model("NER-BiLSTM.h5")

# load saved words index
word_idx = pk.load(open("word_idx.obj", "rb"))

tags = ['B- Geographical Entity',
        'I-Organization',
        'I-Time indicator',
        'I-Geographical Entity',
        'B-Time indicator',
        'I-Artifact',
        'I-Person',
        'B-Organization',
        'B-Artifact',
        'B-Natural Phenomenon',
        'B-Person',
        'I-Natural Phenomenon',
        'B-Geopolitical Entity',
        'I-Geopolitical Entity',
        'I-Event',
        'B-Event',
        'O']


@app.get("/classify/{text}")
async def classify_text_entities(text: str):
    """
        perform named entity recognition on passed text
    """
    word_list = text.strip().split(" ")
    x_new = []
    for word in word_list:
        if word not in word_idx:
            raise HTTPException(
                status_code=400, detail=f"word {word} not in words index(make sure to capitalize the first letter if its a name)!")
        else:
            x_new.append(word_idx[word])

    p = model.predict(np.array([x_new]))
    p = np.argmax(p, axis=-1)
    result = ""
    result += "{:23}{}\n".format("Word", "Prediction")
    result += "-" * 35 + "\n"
    for (w, pred) in zip(range(len(x_new)), p[0]):
        result += "{:20}\t{}\n".format(word_list[w], tags[pred])

    return HTMLResponse(content=result)


@app.get("/")
async def welcome():
    wel = """<html>
                <head>
                    <title>NER BiLSTM API</title>
                </head>
                <body>
                    <p>
                        Welcome to Named Entity Recongnition with Deep Learning API<br>
                        to get started checkout
                        <a href="https://frozen-coast-03690.herokuapp.com/docs">the
                        docs</a><br> <br> <br>
                        Usage example: <br>
                            curl -X GET
                            "https://frozen-coast-03690.herokuapp.com/classify/Ali is
                            swimming" <br><br>
                        Example response:<br> <br>
                            Word  &emsp;  &emsp;  &emsp;  &emsp; &emsp;          Prediction <br>
                            ----------------------------------- <br>
                            Ali &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;     I-Person <br>
                            is &emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;           O <br>
                            swimming &emsp;   &emsp;  &emsp;          O <br>
                        <br><br><br><br><br><br>
                        <h6>Created by Abubakar Yagoub (Blacksuan19)</h6>
                    </p>
                </body>
            </html>
    """
    return HTMLResponse(content=wel, status_code=200)
