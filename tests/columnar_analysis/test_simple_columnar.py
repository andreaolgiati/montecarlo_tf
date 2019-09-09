from montecarlo.rules.columnar.count import Count
from montecarlo.exceptions import RuleEvaluationConditionMet

from faker import Faker

import base64
import datetime
import faker
import io
import json
import numpy as np
import os
import re
import tempfile
import uuid


def gen_preds():
    return Faker().boolean()

def mkdata(data, encoding):
    pred =  gen_preds()
    res = {}
    res['eventVersion'] =  "0"
    res['eventId'] = str(uuid.uuid4())
    res['inferenceTime'] = str(datetime.datetime.now())
    res['capturedData'] = {}
    res['capturedData']['0'] = {}
    res['capturedData']['0']['data'] = data
    res['capturedData']['0']['encoding'] = encoding
    res['capturedData']['1'] = {}
    res['capturedData']['1']['data'] = str({ 'prediction' : pred })
    res['capturedData']['1']['encoding'] = 'json'
    #res['capturedData']['2'] = {}
    #res['capturedData']['2']['data'] = str(preds)
    #res['capturedData']['2']['encoding'] = 'json'

    return res


def create_synthetic_columns(types=[]):
    faker = Faker()
    row = []
    for _type in types:
        val = getattr(faker, _type)()
        row.append(val)
    return row

def create_synthetic_traces(count, column_types):
    _reslist = []
    for _i in range(count):
        cols = create_synthetic_columns( column_types )
        res = mkdata(data=cols, encoding='text/data')
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


class EmailDomainAndStateCount(Count):
    def filter_fn(self, no, data):   
        #print( type(data) )
        if no == 0:          
            email_address = data[2]
            email_domain = re.sub( r'.*@', '@', email_address )
            state = data[5].lower()
            return [state, email_domain]
        else:
            return data
        #elif no == 1:
        #    print( "D=", type(data) )
        #    return data['prediction']


def test_average():

    # Data generation
    tracefile = create_synthetic_traces(count=100, 
                                        column_types=['first_name', 'last_name', 
                                                      'free_email', 'street_address', 'city', 
                                                      'state_abbr', 'zipcode', 
                                                      'random_int'])
    print( tracefile )
    # Executed as part of the platform
    traces = load_data(tracefile)
    # Instantiated as part of the container
    rule = EmailDomainAndStateCount("test_endpoint", 0.1, 0.9)

    try:
        rule.invoke_at_period(start_time=None, end_time=None, traces=traces)
    except RuleEvaluationConditionMet:
        pass
    finally:
        #os.remove(tracefile)
        print( tracefile )
        pass


    #try:
    #    rule.invoke_at_period(start_time=0, end_time=2, traces=traces)
    #except RuleEvaluationConditionMet:
    #    pass
    #finally:
    #    os.remove(tracefile)

test_average()