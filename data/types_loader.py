# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.keras as kr

import json

PoemTypes = []

with open('poems-db/poems-category.json', 'r') as f:
    # 逐行解析json
    for line in f:
        line_json = json.loads(line)
        PoemTypes.append(line_json["name"])

np.savetxt('poems/poems-types.txt', PoemTypes, '%s')
