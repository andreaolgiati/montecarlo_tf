import argparse
import base64
import datetime
import glob
import json
import numpy as np
import random
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs='+', type=str)
args = parser.parse_args()

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def gen_preds():
    res = list(softmax(np.random.normal(size=(10))))
    argmax_res = np.argmax(res)
    return res, argmax_res

def mkdata(data, encoding):
    preds, amax =  gen_preds()
    res = {}
    res['eventVersion'] =  "0"
    res['eventId'] = str(uuid.uuid4())
    res['inferenceTime'] = str(datetime.datetime.now())
    res['capturedData'] = {}
    res['capturedData']['0'] = {}
    res['capturedData']['0']['data'] = data
    res['capturedData']['0']['encoding'] = encoding
    res['capturedData']['1'] = {}
    res['capturedData']['1']['data'] = str({ 'prediction' : amax })
    res['capturedData']['1']['encoding'] = 'json'
    res['capturedData']['2'] = {}
    res['capturedData']['2']['data'] = str(preds)
    res['capturedData']['2']['encoding'] = 'json'

    return res


_reslist = []
for _, fname in enumerate(args.files):
    try:
        with open(fname, "rb") as data_file:
            data = base64.b64encode(data_file.read()).decode()
            res = mkdata(data=data, encoding='base64')
            _reslist.append(res)
    except:
        continue

print(json.dumps(_reslist, indent=True))
