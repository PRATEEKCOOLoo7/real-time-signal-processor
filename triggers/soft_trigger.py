"""Soft trigger engine for lifecycle stage transitions.

Instead of hard rules ("if email opened 3x → convert to Lead"),
uses probabilistic confidence scoring from the scoring engine.
A contact converts to Lead when engagement AND qualification
signals cross configurable thresholds simultaneously.

This reduces false positive conversions compared to rule-based
triggers while catching more genuine buying intent signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from scoring.engine import FullScore

log = logging.getLogger(__name__)


@dataclass
class TriggerRule:
    name: str
    source_stage: str
    target_stage: str
    contact_score_min: float = 0.0
    bant_score_min: float = 0.0
    close_prob_min: float = 0.0
    health_score_min: float = 0.0
    cooldown_hours: float = 24.0


@dataclass
class TriggerResult:
    fired: bool
    rule_name: str = ""
    source_stage: str = ""
    target_stage: str = ""
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    blocked_reason: str = ""


DEFAULT_RULES = [
    TriggerRule(
        name="contact_to_lead",
        source_stage="contact",
        target_stage="lead",
        contact_score_min=50.0,
        bant_score_min=30.0,
        cooldown_hours=48.0,
    ),
    TriggerRule(
        name="lead_to_opportunity",
        source_stage="lead",
        target_stage="opportunity",
        contact_score_min=70.0,
        bant_score_min=60.0,
        close_prob_min=0.3,
        cooldown_hours=72.0,
    ),
    TriggerRule(
        name="flag_at_risk",
        source_stage="customer",
        target_stage="at_risk",
        health_score_min=0.0,  # trigger when health is LOW
        cooldown_hours=168.0,  # weekly
    ),
]


class SoftTrigger:
    """Evaluates whether a contact should transition lifecycle stages."""

    def __init__(self, rules: list[TriggerRule] = None):
        self.rules = rules or DEFAULT_RULES
        self._last_fired: dict[str, datetime] = {}

    def evaluate(self, contact_id: str, current_stage: str,
                 score: FullScore) -> TriggerResult:
        """Check if any trigger rules fire for this contact's scores."""
        applicable = [r for r in self.rules if r.source_stage == current_stage]

        for rule in applicable:
            # Cooldown check
            cooldown_key = f"{contact_id}:{rule.name}"
            last = self._last_fired.get(cooldown_key)
            if last:
                hours_since = (datetime.utcnow() - last).total_seconds() / 3600
                if hours_since < rule.cooldown_hours:
                    continue

            # Evaluate conditions
            reasons = []
            blocked = False

            if rule.name == "flag_at_risk":
                # Special: fires when health is LOW
                if score.health_score.value < 40:
                    reasons.append(f"health_score={score.health_score.value:.1f} (below 40)")
                if score.health_score.trend == "declining":
                    reasons.append(f"trend=declining")
                if not reasons:
                    continue  # health is fine, don't fire
            else:
                cs = score.contact_score.value
                bs = score.bant_score.value
                cp = score.close_prob.value
                hs = score.health_score.value

                if cs < rule.contact_score_min:
                    blocked = True
                    continue
                reasons.append(f"contact_score={cs:.1f}>={rule.contact_score_min}")

                if bs < rule.bant_score_min:
                    blocked = True
                    continue
                reasons.append(f"bant_score={bs:.1f}>={rule.bant_score_min}")

                if rule.close_prob_min > 0 and cp < rule.close_prob_min:
                    blocked = True
                    continue
                if rule.close_prob_min > 0:
                    reasons.append(f"close_prob={cp:.3f}>={rule.close_prob_min}")

            # Confidence based on how far above thresholds
            if rule.name == "flag_at_risk":
                confidence = 1.0 - score.health_score.value / 100
            else:
                margins = [
                    (score.contact_score.value - rule.contact_score_min) / max(rule.contact_score_min, 1),
                    (score.bant_score.value - rule.bant_score_min) / max(rule.bant_score_min, 1),
                ]
                confidence = min(1.0, sum(margins) / len(margins) + 0.5)

            # Fire the trigger
            self._last_fired[cooldown_key] = datetime.utcnow()
            log.info(
                f"TRIGGER FIRED: {rule.name} for {contact_id} "
                f"({rule.source_stage} → {rule.target_stage}) "
                f"confidence={confidence:.3f}"
            )

            return TriggerResult(
                fired=True, rule_name=rule.name,
                source_stage=rule.source_stage,
                target_stage=rule.target_stage,
                confidence=round(confidence, 4),
                reasons=reasons,
            )

        return TriggerResult(fired=False, blocked_reason="no applicable rules met thresholds")
