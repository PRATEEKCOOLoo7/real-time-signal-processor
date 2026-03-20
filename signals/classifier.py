"""Signal classifier — determines intent, sentiment, and buying stage
from raw interaction events.

In production the intent and sentiment models would be fine-tuned
transformers. Here we use rule-based heuristics that capture the
same signal taxonomy, enabling the full pipeline to run without
GPU inference.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class Intent(str, Enum):
    EVALUATING = "evaluating"
    INTERESTED = "interested"
    READY_TO_BUY = "ready_to_buy"
    CHURNING = "churning"
    PASSIVE = "passive"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class BuyingStage(str, Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    DECISION = "decision"
    PURCHASE = "purchase"
    RETENTION = "retention"


# Event type → signal weight mapping
INTENT_SIGNALS = {
    # High intent
    ("web", "pricing_page"): (Intent.EVALUATING, 0.8),
    ("web", "demo_requested"): (Intent.READY_TO_BUY, 0.9),
    ("form", "submitted"): (Intent.INTERESTED, 0.7),
    ("email", "replied"): (Intent.INTERESTED, 0.7),
    ("call", "completed"): (Intent.INTERESTED, 0.6),
    # Medium intent
    ("email", "clicked"): (Intent.EVALUATING, 0.5),
    ("web", "visited"): (Intent.PASSIVE, 0.3),
    ("web", "blog_read"): (Intent.EVALUATING, 0.4),
    ("linkedin", "connection_accepted"): (Intent.INTERESTED, 0.4),
    ("linkedin", "profile_viewed"): (Intent.PASSIVE, 0.2),
    # Low/negative
    ("email", "opened"): (Intent.PASSIVE, 0.2),
    ("email", "bounced"): (Intent.CHURNING, 0.1),
    ("call", "voicemail"): (Intent.PASSIVE, 0.15),
    ("web", "feature_used"): (Intent.INTERESTED, 0.5),
}

STAGE_PROGRESSION = {
    "pricing_page": BuyingStage.DECISION,
    "demo_requested": BuyingStage.DECISION,
    "submitted": BuyingStage.CONSIDERATION,
    "replied": BuyingStage.CONSIDERATION,
    "completed": BuyingStage.CONSIDERATION,
    "clicked": BuyingStage.CONSIDERATION,
    "blog_read": BuyingStage.AWARENESS,
    "visited": BuyingStage.AWARENESS,
    "opened": BuyingStage.AWARENESS,
    "feature_used": BuyingStage.RETENTION,
}


@dataclass
class ClassifiedSignal:
    event_id: str
    contact_id: str
    channel: str
    event_type: str
    intent: Intent
    intent_weight: float
    sentiment: Sentiment
    buying_stage: BuyingStage
    timestamp: str


class SignalClassifier:
    """Classifies raw interaction events into structured signals."""

    def classify(self, event: dict[str, Any]) -> ClassifiedSignal:
        channel = event.get("channel", "")
        etype = event.get("event_type", "")
        props = event.get("properties", {})

        # Intent classification
        key = (channel, etype)
        intent, weight = INTENT_SIGNALS.get(key, (Intent.PASSIVE, 0.1))

        # Boost weight for high-value page visits
        if channel == "web" and props.get("page") in ("/pricing", "/demo"):
            weight = min(weight + 0.2, 1.0)
            intent = Intent.EVALUATING

        # Boost for long call durations
        if channel == "call" and props.get("duration_min", 0) > 20:
            weight = min(weight + 0.15, 1.0)
            intent = Intent.INTERESTED

        # Sentiment from call data
        sentiment = Sentiment.NEUTRAL
        if channel == "call" and "sentiment" in props:
            sentiment = Sentiment(props["sentiment"])
        elif etype in ("replied", "demo_requested", "connection_accepted"):
            sentiment = Sentiment.POSITIVE
        elif etype in ("bounced", "unsubscribed"):
            sentiment = Sentiment.NEGATIVE

        # Buying stage
        stage = STAGE_PROGRESSION.get(etype, BuyingStage.AWARENESS)

        return ClassifiedSignal(
            event_id=event.get("event_id", ""),
            contact_id=event.get("contact_id", ""),
            channel=channel,
            event_type=etype,
            intent=intent,
            intent_weight=round(weight, 3),
            sentiment=sentiment,
            buying_stage=stage,
            timestamp=event.get("timestamp", ""),
        )

    def classify_batch(self, events: list[dict]) -> list[ClassifiedSignal]:
        return [self.classify(e) for e in events]
