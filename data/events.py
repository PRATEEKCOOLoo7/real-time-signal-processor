"""Simulated multi-channel interaction events.

In production these stream from Kafka/Redis consuming webhooks
from email providers (SendGrid), CRM (Salesforce), call platforms
(Gong/Dialpad), web analytics (Segment), and LinkedIn APIs.

Each event represents a single contact interaction with:
- channel: where it happened
- event_type: what happened
- contact_id: who it happened to
- properties: channel-specific metadata
"""

from datetime import datetime, timedelta
import random

CONTACTS = {
    "c_sarah_chen": {
        "name": "Sarah Chen", "company": "Acme Corp",
        "title": "VP of Product", "industry": "fintech",
        "lifecycle_stage": "lead",
    },
    "c_david_kim": {
        "name": "David Kim", "company": "Apex Ventures",
        "title": "Managing Partner", "industry": "venture_capital",
        "lifecycle_stage": "contact",
    },
    "c_rachel_torres": {
        "name": "Rachel Torres", "company": "Meridian Financial",
        "title": "CTO", "industry": "wealth_management",
        "lifecycle_stage": "opportunity",
    },
    "c_james_wright": {
        "name": "James Wright", "company": "NovaTech Solutions",
        "title": "Director of Engineering", "industry": "saas",
        "lifecycle_stage": "customer",
    },
    "c_maria_garcia": {
        "name": "Maria Garcia", "company": "Pinnacle Analytics",
        "title": "Head of Data", "industry": "analytics",
        "lifecycle_stage": "contact",
    },
}

# Realistic interaction history
def generate_events(contact_id: str, days_back: int = 30) -> list[dict]:
    """Generate a realistic sequence of interaction events for a contact."""
    now = datetime.utcnow()
    events = []
    contact = CONTACTS.get(contact_id, {})
    stage = contact.get("lifecycle_stage", "contact")

    # Event templates by engagement level
    if stage == "customer":
        templates = [
            {"channel": "email", "event_type": "opened", "weight": 8},
            {"channel": "email", "event_type": "clicked", "weight": 5},
            {"channel": "web", "event_type": "visited", "weight": 6},
            {"channel": "call", "event_type": "completed", "weight": 3},
            {"channel": "web", "event_type": "feature_used", "weight": 7},
            {"channel": "email", "event_type": "replied", "weight": 2},
        ]
    elif stage == "opportunity":
        templates = [
            {"channel": "email", "event_type": "opened", "weight": 7},
            {"channel": "email", "event_type": "clicked", "weight": 6},
            {"channel": "email", "event_type": "replied", "weight": 4},
            {"channel": "call", "event_type": "completed", "weight": 5},
            {"channel": "call", "event_type": "voicemail", "weight": 2},
            {"channel": "web", "event_type": "pricing_page", "weight": 4},
            {"channel": "web", "event_type": "demo_requested", "weight": 1},
            {"channel": "linkedin", "event_type": "profile_viewed", "weight": 3},
        ]
    elif stage == "lead":
        templates = [
            {"channel": "email", "event_type": "opened", "weight": 5},
            {"channel": "email", "event_type": "clicked", "weight": 3},
            {"channel": "web", "event_type": "visited", "weight": 4},
            {"channel": "web", "event_type": "blog_read", "weight": 3},
            {"channel": "form", "event_type": "submitted", "weight": 1},
            {"channel": "linkedin", "event_type": "connection_accepted", "weight": 1},
        ]
    else:  # contact
        templates = [
            {"channel": "email", "event_type": "opened", "weight": 2},
            {"channel": "web", "event_type": "visited", "weight": 2},
            {"channel": "email", "event_type": "bounced", "weight": 1},
        ]

    total_weight = sum(t["weight"] for t in templates)
    num_events = random.randint(5, 25)

    for i in range(num_events):
        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0
        selected = templates[0]
        for t in templates:
            cumulative += t["weight"]
            if r <= cumulative:
                selected = t
                break

        ts = now - timedelta(
            days=random.uniform(0, days_back),
            hours=random.uniform(0, 23),
            minutes=random.uniform(0, 59),
        )

        event = {
            "event_id": f"evt_{contact_id}_{i:03d}",
            "contact_id": contact_id,
            "channel": selected["channel"],
            "event_type": selected["event_type"],
            "timestamp": ts.isoformat(),
            "properties": _event_properties(selected["channel"], selected["event_type"]),
        }
        events.append(event)

    events.sort(key=lambda e: e["timestamp"])
    return events


def _event_properties(channel: str, event_type: str) -> dict:
    if channel == "email":
        return {
            "subject": random.choice([
                "AI-Powered Portfolio Analytics",
                "Your Competitive Intelligence Brief",
                "Quick question about your tech stack",
                "Follow-up: Revenue Intelligence Demo",
            ]),
            "campaign": random.choice(["nurture_q4", "product_launch", "reengagement"]),
        }
    elif channel == "web":
        return {
            "page": random.choice([
                "/pricing", "/product", "/blog/ai-revenue-ops",
                "/demo", "/case-studies", "/about",
            ]),
            "duration_sec": random.randint(10, 300),
            "referrer": random.choice(["google", "linkedin", "direct", "email"]),
        }
    elif channel == "call":
        return {
            "duration_min": random.randint(2, 45),
            "sentiment": random.choice(["positive", "neutral", "negative"]),
            "topics": random.sample(
                ["pricing", "competition", "timeline", "technical", "budget"],
                k=random.randint(1, 3),
            ),
        }
    elif channel == "linkedin":
        return {"action": event_type}
    elif channel == "form":
        return {
            "form_name": random.choice(["demo_request", "whitepaper", "newsletter"]),
        }
    return {}


def get_all_events() -> dict[str, list[dict]]:
    """Generate events for all contacts."""
    return {cid: generate_events(cid) for cid in CONTACTS}
