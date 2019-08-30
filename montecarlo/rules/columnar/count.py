from collections import Counter
from montecarlo.rules.rule import Rule

import io
import json
import numpy as np
import base64


class Count(Rule):
    def __init__(self, endpoint, min_threshold=0, max_threshold=1.0):
        super().__init__(endpoint, other_periods=None)
        self.min_threshold = int(min_threshold)
        self.max_threshold = int(max_threshold)
        self.logger.info("Count rule created.")

    def filter_fn(self, csv):
        return csv

    def load_trace_data( self, trace ):
        csv = trace['capturedData']['0']['data']
        if type(csv) is not list:
            csv = json.loads(csv)
        filtered_csv = tuple(self.filter_fn(csv))
        return (trace['eventId'], trace['inferenceTime'], filtered_csv)


    def invoke_at_period(self, start_time, end_time, traces, storage_handler=None, **kwargs):
        counters = []
        for trace in traces:
            print(trace)
            _event_id, _event_time, filtered_data = self.load_trace_data(trace)
            if len(counters)==0:
                for i in range(len(filtered_data)):
                    counters.append(Counter())
            for i, item in enumerate(filtered_data):
                counters[i][item] += 1
        #for counter in counters:
        #    if counter[max(counter)] > self.max_threshold:
        #        print( "MAX:", counter )
        #    if counter[min(counter)] < self.min_threshold:
        #        print( "MIN:", counter )
        print( counters )
        
