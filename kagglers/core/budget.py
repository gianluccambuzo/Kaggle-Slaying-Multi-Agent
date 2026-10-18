from __future__ import annotations

from dataclasses import dataclass

from kagglers.core.types import PackRules


@dataclass(frozen=True)
class BudgetLimits:
    llm_usd: float
    compute_minutes: float
    submission_reserve: int = 1


class Budget:
    def __init__(self, limits: BudgetLimits, rules: PackRules) -> None:
        self.limits = limits
        self.rules = rules
        self.llm_spent_usd = 0.0
        self.compute_seconds = 0.0

    def charge_llm(self, usd: float) -> None:
        self.llm_spent_usd += usd

    def charge_compute(self, seconds: float) -> None:
        self.compute_seconds += seconds

    def exhausted(self) -> str | None:
        if self.llm_spent_usd >= self.limits.llm_usd:
            return "llm_budget"
        if self.compute_seconds >= self.limits.compute_minutes * 60:
            return "compute_budget"
        return None

    def can_submit(self, submissions_today: int, final: bool = False) -> bool:
        cap = self.rules.daily_submission_limit
        if not final:
            cap -= self.limits.submission_reserve
        return submissions_today < cap
