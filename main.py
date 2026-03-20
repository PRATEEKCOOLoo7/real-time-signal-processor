"""Real-Time Signal Processor — Demo

Processes multi-channel interaction events, classifies signals,
computes 4 scoring matrices, and evaluates soft triggers for
lifecycle stage transitions.

Run: python main.py
"""

import logging
from data.events import CONTACTS, generate_events
from signals.classifier import SignalClassifier
from scoring.engine import ScoringEngine
from triggers.soft_trigger import SoftTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    classifier = SignalClassifier()
    scorer = ScoringEngine(recency_half_life_days=14.0)
    trigger = SoftTrigger()

    print(f"\n{'='*70}")
    print("  Real-Time Signal Processor")
    print("  Events → Classify → Score (4 matrices) → Soft Trigger")
    print(f"{'='*70}")

    for cid, info in CONTACTS.items():
        events = generate_events(cid, days_back=30)
        signals = classifier.classify_batch(events)
        scores = scorer.score(cid, signals)
        stage = info["lifecycle_stage"]
        trig = trigger.evaluate(cid, stage, scores)

        print(f"\n{'─'*70}")
        print(f"  {info['name']} | {info['title']} @ {info['company']}")
        print(f"  Stage: {stage} | Signals: {len(signals)}")
        print(f"{'─'*70}")

        cs = scores.contact_score
        print(f"  Contact Score:  {cs.value:5.1f}/100 ({cs.tier}) — top channel: {cs.top_channel}")

        bs = scores.bant_score
        bant_str = (f"B={bs.budget:.0f} A={bs.authority:.0f} "
                    f"N={bs.need:.0f} T={bs.timeline:.0f}")
        qual = "QUALIFIED" if bs.qualified else "not qualified"
        print(f"  BANT Score:     {bs.value:5.1f}/100 ({qual}) — {bant_str}")

        cp = scores.close_prob
        print(f"  Close Prob:     {cp.value:5.3f}   — stage={cp.stage_factor:.2f} eng={cp.engagement_factor:.2f} vel={cp.velocity_factor:.2f}")

        hs = scores.health_score
        risk_flag = " ⚠ AT RISK" if hs.at_risk else ""
        print(f"  Health Score:   {hs.value:5.1f}/100 (trend: {hs.trend}){risk_flag}")
        print(f"                  usage={hs.usage_score:.0f} sentiment={hs.sentiment_score:.0f} engagement={hs.engagement_score:.0f}")

        if trig.fired:
            print(f"\n  🔔 TRIGGER: {trig.rule_name}")
            print(f"     {trig.source_stage} → {trig.target_stage} (confidence: {trig.confidence:.3f})")
            for r in trig.reasons:
                print(f"       {r}")
        else:
            print(f"\n  No trigger fired ({trig.blocked_reason})")

    print(f"\n{'='*70}")
    print("  Processing complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
