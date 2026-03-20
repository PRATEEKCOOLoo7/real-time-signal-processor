"""Four scoring matrices for revenue intelligence.

Each contact gets scored across 4 dimensions, updated on every
new signal. The scores drive agent actions and lifecycle transitions.

1. Contact Score: engagement level (email opens, clicks, web visits)
2. BANT Score: qualification signals (budget, authority, need, timeline)
3. Close Probability: likelihood of deal closure
4. Health Score: customer health for retention (usage, sentiment, support)
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from signals.classifier import ClassifiedSignal, Intent, Sentiment, BuyingStage

log = logging.getLogger(__name__)


@dataclass
class ContactScore:
    """Engagement-based contact score (0-100)."""
    value: float = 0.0
    signals_counted: int = 0
    top_channel: str = ""
    last_activity: str = ""

    @property
    def tier(self) -> str:
        if self.value >= 80:
            return "hot"
        elif self.value >= 50:
            return "warm"
        elif self.value >= 20:
            return "cool"
        return "cold"


@dataclass
class BANTScore:
    """BANT qualification score (0-100)."""
    value: float = 0.0
    budget: float = 0.0
    authority: float = 0.0
    need: float = 0.0
    timeline: float = 0.0

    @property
    def qualified(self) -> bool:
        return self.value >= 60 and sum(
            1 for s in [self.budget, self.authority, self.need, self.timeline]
            if s >= 50
        ) >= 3


@dataclass
class CloseProb:
    """Opportunity close probability (0-1)."""
    value: float = 0.0
    stage_factor: float = 0.0
    engagement_factor: float = 0.0
    velocity_factor: float = 0.0


@dataclass
class HealthScore:
    """Customer health score (0-100)."""
    value: float = 50.0
    usage_score: float = 50.0
    sentiment_score: float = 50.0
    engagement_score: float = 50.0
    trend: str = "stable"  # improving, stable, declining

    @property
    def at_risk(self) -> bool:
        return self.value < 40 or self.trend == "declining"


@dataclass
class FullScore:
    """All 4 scoring dimensions for a contact."""
    contact_id: str
    contact_score: ContactScore
    bant_score: BANTScore
    close_prob: CloseProb
    health_score: HealthScore
    scored_at: str = ""
    model_version: str = "v1.0"


# Weights for contact scoring by event type
ENGAGEMENT_WEIGHTS = {
    ("email", "opened"): 2,
    ("email", "clicked"): 5,
    ("email", "replied"): 15,
    ("email", "bounced"): -3,
    ("web", "visited"): 3,
    ("web", "pricing_page"): 12,
    ("web", "demo_requested"): 20,
    ("web", "blog_read"): 4,
    ("web", "feature_used"): 6,
    ("call", "completed"): 10,
    ("call", "voicemail"): 1,
    ("form", "submitted"): 15,
    ("linkedin", "connection_accepted"): 8,
    ("linkedin", "profile_viewed"): 2,
}

# BANT signal indicators
BANT_SIGNALS = {
    "budget": {
        ("call", "completed"): 15,  # calls often discuss budget
        ("web", "pricing_page"): 20,
        ("form", "submitted"): 10,
    },
    "authority": {
        ("call", "completed"): 20,  # direct conversation = authority signal
        ("email", "replied"): 15,
        ("linkedin", "connection_accepted"): 10,
    },
    "need": {
        ("web", "demo_requested"): 25,
        ("web", "pricing_page"): 15,
        ("web", "blog_read"): 5,
        ("form", "submitted"): 20,
    },
    "timeline": {
        ("web", "demo_requested"): 20,
        ("email", "replied"): 10,
        ("call", "completed"): 15,
    },
}


class ScoringEngine:
    """Computes all 4 scoring matrices from classified signals."""

    def __init__(self, recency_half_life_days: float = 14.0):
        self.half_life = recency_half_life_days

    def score(self, contact_id: str, signals: list[ClassifiedSignal]) -> FullScore:
        if not signals:
            return FullScore(
                contact_id=contact_id,
                contact_score=ContactScore(),
                bant_score=BANTScore(),
                close_prob=CloseProb(),
                health_score=HealthScore(),
                scored_at=datetime.utcnow().isoformat(),
            )

        contact = self._score_contact(signals)
        bant = self._score_bant(signals)
        close = self._score_close(signals, contact, bant)
        health = self._score_health(signals)

        return FullScore(
            contact_id=contact_id,
            contact_score=contact,
            bant_score=bant,
            close_prob=close,
            health_score=health,
            scored_at=datetime.utcnow().isoformat(),
        )

    def _score_contact(self, signals: list[ClassifiedSignal]) -> ContactScore:
        """Engagement scoring with recency decay."""
        now = datetime.utcnow()
        total = 0.0
        channels = Counter()

        for sig in signals:
            key = (sig.channel, sig.event_type)
            base_weight = ENGAGEMENT_WEIGHTS.get(key, 1)

            # Recency decay: recent events count more
            try:
                ts = datetime.fromisoformat(sig.timestamp)
                days_ago = (now - ts).total_seconds() / 86400
            except (ValueError, TypeError):
                days_ago = 15

            decay = math.exp(-0.693 * days_ago / self.half_life)  # 0.693 = ln(2)
            total += base_weight * decay
            channels[sig.channel] += 1

        # Normalize to 0-100
        value = min(100.0, total * 2.5)
        top_channel = channels.most_common(1)[0][0] if channels else ""
        last = signals[-1].timestamp if signals else ""

        return ContactScore(
            value=round(value, 1),
            signals_counted=len(signals),
            top_channel=top_channel,
            last_activity=last,
        )

    def _score_bant(self, signals: list[ClassifiedSignal]) -> BANTScore:
        """BANT qualification scoring from interaction signals."""
        dims = {"budget": 0.0, "authority": 0.0, "need": 0.0, "timeline": 0.0}

        for sig in signals:
            key = (sig.channel, sig.event_type)
            for dim_name, mapping in BANT_SIGNALS.items():
                if key in mapping:
                    dims[dim_name] += mapping[key]

            # Intent-based boosts
            if sig.intent == Intent.READY_TO_BUY:
                dims["need"] += 10
                dims["timeline"] += 15
            elif sig.intent == Intent.EVALUATING:
                dims["need"] += 5
                dims["budget"] += 5

        # Normalize each dim to 0-100
        for k in dims:
            dims[k] = min(100.0, dims[k])

        overall = sum(dims.values()) / 4
        return BANTScore(
            value=round(overall, 1),
            budget=round(dims["budget"], 1),
            authority=round(dims["authority"], 1),
            need=round(dims["need"], 1),
            timeline=round(dims["timeline"], 1),
        )

    def _score_close(self, signals: list[ClassifiedSignal],
                     contact: ContactScore, bant: BANTScore) -> CloseProb:
        """Close probability from stage, engagement, and velocity."""
        # Stage factor
        stages = [s.buying_stage for s in signals]
        stage_weights = {
            BuyingStage.AWARENESS: 0.1,
            BuyingStage.CONSIDERATION: 0.3,
            BuyingStage.DECISION: 0.6,
            BuyingStage.PURCHASE: 0.9,
            BuyingStage.RETENTION: 0.5,
        }
        if stages:
            highest_stage = max(stages, key=lambda s: stage_weights.get(s, 0))
            stage_factor = stage_weights.get(highest_stage, 0.1)
        else:
            stage_factor = 0.05

        # Engagement factor (from contact score)
        eng_factor = contact.value / 100.0

        # Velocity factor: how fast are signals coming in?
        if len(signals) >= 2:
            try:
                first = datetime.fromisoformat(signals[0].timestamp)
                last = datetime.fromisoformat(signals[-1].timestamp)
                span_days = max((last - first).total_seconds() / 86400, 1)
                velocity = len(signals) / span_days
                vel_factor = min(velocity / 3.0, 1.0)  # 3+ events/day = max velocity
            except (ValueError, TypeError):
                vel_factor = 0.3
        else:
            vel_factor = 0.1

        # Weighted combination
        prob = 0.4 * stage_factor + 0.3 * eng_factor + 0.3 * vel_factor
        prob = min(max(prob, 0.0), 1.0)

        return CloseProb(
            value=round(prob, 4),
            stage_factor=round(stage_factor, 3),
            engagement_factor=round(eng_factor, 3),
            velocity_factor=round(vel_factor, 3),
        )

    def _score_health(self, signals: list[ClassifiedSignal]) -> HealthScore:
        """Customer health from usage, sentiment, and engagement recency."""
        now = datetime.utcnow()

        # Usage: feature_used events
        usage_events = [s for s in signals if s.event_type == "feature_used"]
        usage_score = min(100.0, len(usage_events) * 15)

        # Sentiment: average from call sentiment data
        sentiments = [s.sentiment for s in signals if s.sentiment != Sentiment.NEUTRAL]
        if sentiments:
            pos = sum(1 for s in sentiments if s == Sentiment.POSITIVE)
            neg = sum(1 for s in sentiments if s == Sentiment.NEGATIVE)
            total = len(sentiments)
            sent_score = 50 + (pos - neg) / total * 50
        else:
            sent_score = 50.0

        # Engagement recency
        if signals:
            try:
                last_ts = datetime.fromisoformat(signals[-1].timestamp)
                days_since = (now - last_ts).total_seconds() / 86400
                eng_score = max(0, 100 - days_since * 5)  # -5 per day of silence
            except (ValueError, TypeError):
                eng_score = 50.0
        else:
            eng_score = 0.0

        overall = (usage_score * 0.35 + sent_score * 0.35 + eng_score * 0.3)

        # Trend: compare recent vs older signals
        midpoint = len(signals) // 2
        if midpoint > 2:
            recent_intents = [s.intent_weight for s in signals[midpoint:]]
            older_intents = [s.intent_weight for s in signals[:midpoint]]
            recent_avg = sum(recent_intents) / len(recent_intents)
            older_avg = sum(older_intents) / len(older_intents)
            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return HealthScore(
            value=round(overall, 1),
            usage_score=round(usage_score, 1),
            sentiment_score=round(sent_score, 1),
            engagement_score=round(eng_score, 1),
            trend=trend,
        )
