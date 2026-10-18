from kagglers.core.budget import Budget, BudgetLimits
from kagglers.core.types import PackRules


def _rules(limit=5):
    return PackRules("auc", True, limit, False, False)


def _budget(llm=10.0, minutes=1.0, reserve=1, limit=5):
    return Budget(BudgetLimits(llm, minutes, reserve), _rules(limit))


def test_exhausted_none_when_under_limits():
    b = _budget()
    b.charge_llm(1.0)
    b.charge_compute(10)
    assert b.exhausted() is None


def test_exhausted_reports_llm_budget():
    b = _budget(llm=1.0)
    b.charge_llm(1.5)
    assert b.exhausted() == "llm_budget"


def test_exhausted_reports_compute_budget_after_charging_seconds():
    b = _budget(minutes=1)
    b.charge_compute(61)
    assert b.exhausted() == "compute_budget"


def test_can_submit_respects_reserve():
    b = _budget(limit=5, reserve=1)
    assert b.can_submit(3) is True
    assert b.can_submit(4) is False


def test_final_submission_may_use_reserve():
    b = _budget(limit=5, reserve=1)
    assert b.can_submit(4, final=True) is True
    assert b.can_submit(5, final=True) is False
