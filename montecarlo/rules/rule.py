#from montecarlo.exceptions import TensorUnavailable
from montecarlo.core.utils import get_logger
#from montecarlo.analysis.utils import no_refresh
from montecarlo.exceptions import RuleEvaluationConditionMet
from abc import ABC, abstractmethod

logger = get_logger()

"""
class RequiredTensors:
    def __init__(self, period):
        self.period = period
        self.tensor_names = {}
        self.logger = logger
        self.should_match_regex = {}

    def need_tensor(self, name, steps, should_match_regex=False):
        if name not in self.tensor_names:
            self.tensor_names[name] = steps
        else:
            self.tensor_names[name].extend(steps)

        if should_match_regex:
            self.should_match_regex[name] = True

    def _check_if_steps_available(self, tname, steps):
        t = self.period.tensor(tname)
        for st in steps:
            t.value(st)

    # returns number of arrays fetched for this rule
    def _fetch_tensors(self):
        required_steps = set()
        for steps in self.tensor_names.values():
            required_steps = required_steps.union(set(steps))
        required_steps = sorted(required_steps)
        if required_steps:
            self.logger.debug(f"Waiting for required_steps: {required_steps}")
        self.period.wait_for_steps(required_steps)
        self.period.get_tensors(self.tensor_names,
                               should_regex_match=self.should_match_regex)
        for tensorname, steps in self.tensor_names.items():
            # check whether we should match regex for this tensorname
            # False refers to the default value if the key does not exist in the dictionary
            if self.should_match_regex.get(tensorname, False):
                regex = tensorname
                tnames = self.period.tensors_matching_regex([regex])
            else:
                tnames = [tensorname]
            for tname in tnames:
                if not self.period.has_tensor(tname):
                    raise TensorUnavailable(tensorname)
                else:
                    self._check_if_steps_available(tname, steps)
"""


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

    """
    # step here is global step
    @abstractmethod
    def invoke_at_period(self, start_time, stop_time, traces, storage_handler=None, **kwargs):
        # implementation check for tensor
        # do checkpoint if needed at periodic interval --> storage_handler.save("last_processed_tensor",(tensorname,step))
        # checkpoiniting is needed if execution is longer duration, so that we don't
        # lose the work done in certain step
        pass
    """

    # step specific for which global step this rule was invoked
    # storage_handler is used to save & get states across different invocations
    def invoke(self, period, traces, storage_handler=None, **kwargs):
        self.logger.debug('Invoking rule {} for step {}'.format(self.__class__.__name__, period))
        self.endpoint.wait_for_steps([period])
        req_tensors_requests = self.required_tensors(period)
        self._fetch_tensors_for_periods(req_tensors_requests)

        # do not refresh during invoke at step since required tensors are already here
        with no_refresh(self.periods):
            val = self.invoke_at_step(period)

        if val:
            self.run_actions()
            raise RuleEvaluationConditionMet

    def register_action(self, actions):
        self.actions = actions

    def run_actions(self):
        if self.actions is not None:
            for action in self.actions:
                action.run(rule_name=self.__class__.__name__)
