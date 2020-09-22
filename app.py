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
from fastapi import FastAPI

fapi = FastAPI()

# load saved model
model = tf.keras.models.load_model("NER-BiLSTM.h5")

# load saved words index
word_idx = pk.load(open("word_idx.obj", "rb"))

# word tags (TODO: convert to human readable format)
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


@fapi.get("/predict/{text}")
async def create_test_input_from_text(text):
    word_list = text.split(" ")
    x_new = []
    for word in word_list:
        x_new.append(word_idx[word])

    p = model.predict(np.array([x_new]))
    p = np.argmax(p, axis=-1)
    result = ""
    result += "{:20}\t{}\n".format("Word", "Prediction")
    result += "-" * 35

    for (w, pred) in zip(range(len(x_new)), p[0]):
        result += "{:20}\t{}".format(word_list[w], tags[pred])

    return result
