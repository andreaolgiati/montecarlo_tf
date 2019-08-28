class NoMoreData(Exception):
  pass


class RuleEvaluationConditionMet(Exception):
  def __str__(self):
    return 'Rule evaluation resulted in the condition being met'
