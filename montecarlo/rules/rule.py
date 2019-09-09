#from montecarlo.exceptions import TensorUnavailable
from montecarlo.core.utils import get_logger
#from montecarlo.analysis.utils import no_refresh
from montecarlo.exceptions import RuleEvaluationConditionMet
from abc import ABC, abstractmethod

import json

logger = get_logger()


# This is Rule interface
class Rule(ABC):
    def __init__(self, endpoint, other_periods=None):
        self.endpoint = endpoint
        self.other_periods = other_periods

        self.periods = [endpoint]
        if self.other_periods is not None:
            self.periods += [x for x in self.other_periods]

        self.actions = None
        self.logger = logger
        pass

    def load_trace_data( self, trace ):
        res = []
        trace_nos = sorted(trace['capturedData'].keys(), key=int)
        
        for trace_no in trace_nos:
            data = trace['capturedData'][trace_no]['data']
            j_data = self.filter_fn(int(trace_no), data)
            res.append(j_data)
        return (trace['eventId'], trace['inferenceTime'], res )
