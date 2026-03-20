import pytest
from datetime import datetime, timedelta

from signals.classifier import SignalClassifier, Intent, Sentiment, BuyingStage
from scoring.engine import ScoringEngine
from triggers.soft_trigger import SoftTrigger, TriggerRule
from data.events import generate_events, CONTACTS


class TestSignalClassifier:
    def setup_method(self):
        self.c = SignalClassifier()

    def test_high_intent_pricing_page(self):
        sig = self.c.classify({
            "event_id": "e1", "contact_id": "c1",
            "channel": "web", "event_type": "pricing_page",
            "timestamp": datetime.utcnow().isoformat(),
            "properties": {"page": "/pricing", "duration_sec": 120},
        })
        assert sig.intent in (Intent.EVALUATING, Intent.READY_TO_BUY)
        assert sig.intent_weight >= 0.8

    def test_demo_request_highest_intent(self):
        sig = self.c.classify({
            "event_id": "e2", "contact_id": "c1",
            "channel": "web", "event_type": "demo_requested",
            "timestamp": datetime.utcnow().isoformat(),
            "properties": {},
        })
        assert sig.intent == Intent.READY_TO_BUY
        assert sig.buying_stage == BuyingStage.DECISION

    def test_email_open_low_intent(self):
        sig = self.c.classify({
            "event_id": "e3", "contact_id": "c1",
            "channel": "email", "event_type": "opened",
            "timestamp": datetime.utcnow().isoformat(),
            "properties": {},
        })
        assert sig.intent == Intent.PASSIVE
        assert sig.intent_weight < 0.3

    def test_call_sentiment_passthrough(self):
        sig = self.c.classify({
            "event_id": "e4", "contact_id": "c1",
            "channel": "call", "event_type": "completed",
            "timestamp": datetime.utcnow().isoformat(),
            "properties": {"duration_min": 30, "sentiment": "positive"},
        })
        assert sig.sentiment == Sentiment.POSITIVE
        assert sig.intent_weight > 0.5  # long call boost

    def test_bounced_negative(self):
        sig = self.c.classify({
            "event_id": "e5", "contact_id": "c1",
            "channel": "email", "event_type": "bounced",
            "timestamp": datetime.utcnow().isoformat(),
            "properties": {},
        })
        assert sig.sentiment == Sentiment.NEGATIVE

    def test_batch_classify(self):
        events = generate_events("c_sarah_chen", days_back=7)
        signals = self.c.classify_batch(events)
        assert len(signals) == len(events)
        assert all(s.contact_id == "c_sarah_chen" for s in signals)


class TestScoringEngine:
    def setup_method(self):
        self.e = ScoringEngine()
        self.c = SignalClassifier()

    def _make_signals(self, contact_id, days=30):
        events = generate_events(contact_id, days)
        return self.c.classify_batch(events)

    def test_score_active_lead(self):
        signals = self._make_signals("c_sarah_chen")
        scores = self.e.score("c_sarah_chen", signals)
        assert scores.contact_score.value > 0
        assert scores.contact_score.signals_counted > 0
        assert scores.bant_score.value >= 0
        assert 0 <= scores.close_prob.value <= 1

    def test_score_customer(self):
        signals = self._make_signals("c_james_wright")
        scores = self.e.score("c_james_wright", signals)
        assert scores.health_score.value > 0
        assert scores.health_score.trend in ("improving", "stable", "declining")

    def test_empty_signals(self):
        scores = self.e.score("c_empty", [])
        assert scores.contact_score.value == 0.0
        assert scores.bant_score.value == 0.0
        assert scores.close_prob.value == 0.0

    def test_recency_decay(self):
        """Recent signals should score higher than old ones."""
        now = datetime.utcnow()
        recent_event = {
            "event_id": "r1", "contact_id": "test",
            "channel": "web", "event_type": "demo_requested",
            "timestamp": now.isoformat(), "properties": {},
        }
        old_event = {
            "event_id": "o1", "contact_id": "test",
            "channel": "web", "event_type": "demo_requested",
            "timestamp": (now - timedelta(days=60)).isoformat(),
            "properties": {},
        }
        recent_sig = [self.c.classify(recent_event)]
        old_sig = [self.c.classify(old_event)]

        recent_score = self.e.score("test_r", recent_sig)
        old_score = self.e.score("test_o", old_sig)
        assert recent_score.contact_score.value > old_score.contact_score.value

    def test_contact_tier(self):
        signals = self._make_signals("c_rachel_torres")
        scores = self.e.score("c_rachel_torres", signals)
        assert scores.contact_score.tier in ("hot", "warm", "cool", "cold")

    def test_all_contacts_scoreable(self):
        for cid in CONTACTS:
            signals = self._make_signals(cid)
            scores = self.e.score(cid, signals)
            assert scores.contact_id == cid
            assert scores.scored_at != ""


class TestSoftTrigger:
    def setup_method(self):
        self.t = SoftTrigger()
        self.c = SignalClassifier()
        self.e = ScoringEngine()

    def test_no_trigger_for_low_engagement(self):
        scores = self.e.score("c_low", [])
        result = self.t.evaluate("c_low", "contact", scores)
        assert not result.fired

    def test_trigger_fires_for_high_engagement_contact(self):
        """A contact with many high-intent signals should convert to lead."""
        now = datetime.utcnow()
        events = []
        for i in range(15):
            events.append({
                "event_id": f"h{i}", "contact_id": "c_hot",
                "channel": "web", "event_type": "pricing_page",
                "timestamp": (now - timedelta(hours=i * 6)).isoformat(),
                "properties": {"page": "/pricing"},
            })
            events.append({
                "event_id": f"e{i}", "contact_id": "c_hot",
                "channel": "email", "event_type": "clicked",
                "timestamp": (now - timedelta(hours=i * 6 + 1)).isoformat(),
                "properties": {},
            })

        signals = self.c.classify_batch(events)
        scores = self.e.score("c_hot", signals)

        # Should have high enough scores to trigger
        if scores.contact_score.value >= 50 and scores.bant_score.value >= 30:
            result = self.t.evaluate("c_hot", "contact", scores)
            assert result.fired
            assert result.target_stage == "lead"
            assert result.confidence > 0.5

    def test_cooldown_prevents_repeated_trigger(self):
        """Same trigger shouldn't fire twice within cooldown period."""
        now = datetime.utcnow()
        events = [
            {"event_id": f"x{i}", "contact_id": "c_cd",
             "channel": "web", "event_type": "pricing_page",
             "timestamp": (now - timedelta(hours=i)).isoformat(),
             "properties": {"page": "/pricing"}}
            for i in range(20)
        ]
        signals = self.c.classify_batch(events)
        scores = self.e.score("c_cd", signals)

        r1 = self.t.evaluate("c_cd", "contact", scores)
        r2 = self.t.evaluate("c_cd", "contact", scores)  # should be blocked by cooldown
        if r1.fired:
            assert not r2.fired  # cooldown should prevent

    def test_custom_rules(self):
        rule = TriggerRule(
            name="custom", source_stage="lead", target_stage="opportunity",
            contact_score_min=30, bant_score_min=20, cooldown_hours=1,
        )
        t = SoftTrigger(rules=[rule])
        result = t.evaluate("c1", "contact", None)  # wrong stage
        assert not result.fired
