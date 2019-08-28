from montecarlo.rules.image.luminance import Luminance
from montecarlo.exceptions import RuleEvaluationConditionMet
from PIL import Image

import base64
import datetime
import io
import json
import numpy as np
import os
import tempfile
import uuid



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


def create_synthetic_img(x,y, avg_lumi=0, format='png'):
    a = np.random.rand(x,y,3) * 255
    image = Image.fromarray(a.astype('uint8')).convert('RGBA')
    with io.BytesIO() as output:
        image.save(output, format=format)
        contents = output.getvalue()
    return contents

def create_synthetic_traces(avg_lumi):
    _reslist = []
    for _i in range(10):
        jpg_bytes = create_synthetic_img( 100, 100, avg_lumi )
        jpg_string = base64.b64encode(jpg_bytes).decode('ascii')
        res = mkdata(data=jpg_string, encoding='base64')
        _reslist.append(res)
    payload = json.dumps(_reslist, indent=True)
    
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(payload)
    finally:
        pass
        #print( path )
    return path


def load_data(file_name):
    with open(file_name) as f:
        captures = json.loads(f.read())
    return captures


def test_luminance():

    # Data generation
    tracefile = create_synthetic_traces(avg_lumi=0.3)
    # Executed as part of the platform
    traces = load_data(tracefile)
    # Instantiated as part of the container
    rule = Luminance("test_endpoint", 0.1, 0.9)

    try:
        rule.invoke_at_period(start_time=None, end_time=None, traces=traces)
    except:
        os.remove(tracefile)


    try:
        rule.invoke_at_period(start_time=0, end_time=2, traces=traces)
    except RuleEvaluationConditionMet:
        pass
    finally:
        os.remove(tracefile)

