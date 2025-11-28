"""Simple Kafka-like streaming simulator for ESG events.

This module does not require a running Kafka cluster. It mimics
producer/consumer APIs and allows you to demonstrate real-time ESG
scoring in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import queue
import threading
import time
import random


@dataclass
class ESGEvent:
    """Represents a streaming ESG-relevant event.

    Examples: new satellite tile, news sentiment update, disclosure.
    """

    company_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: float


class InMemoryTopic:
    """Thread-safe in-memory topic behaving like a Kafka partition."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: "queue.Queue[ESGEvent]" = queue.Queue(maxsize=maxsize)

    def put(self, event: ESGEvent) -> None:
        self._q.put(event)

    def get(self, timeout: Optional[float] = None) -> ESGEvent:
        return self._q.get(timeout=timeout)

    def empty(self) -> bool:
        return self._q.empty()


class ESGProducer:
    """Simulated event producer."""

    def __init__(self, topic: InMemoryTopic) -> None:
        self.topic = topic

    def send(self, company_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        ev = ESGEvent(company_id=company_id, event_type=event_type,
                      payload=payload, timestamp=time.time())
        self.topic.put(ev)


class ESGConsumer:
    """Simple blocking consumer over the in-memory topic."""

    def __init__(self, topic: InMemoryTopic) -> None:
        self.topic = topic

    def iterate(self, max_events: Optional[int] = None, timeout: float = 0.1) -> Iterable[ESGEvent]:
        count = 0
        while True:
            if max_events is not None and count >= max_events:
                break
            try:
                ev = self.topic.get(timeout=timeout)
            except queue.Empty:
                break
            count += 1
            yield ev


def start_background_producer(
    topic: InMemoryTopic,
    company_ids: List[str],
    event_rate_hz: float = 2.0,
    stop_after: float = 5.0,
) -> threading.Thread:
    """Spawn a background thread that produces random ESG events.

    This is analogous to a real Kafka producer connected to upstream
    systems such as satellite tiling jobs or NLP pipelines.
    """

    def _run() -> None:
        end_time = time.time() + stop_after
        prod = ESGProducer(topic)
        while time.time() < end_time:
            cid = random.choice(company_ids)
            etype = random.choice(
                ["satellite_update", "news_update", "disclosure"])
            payload: Dict[str, Any]
            if etype == "satellite_update":
                payload = {"ndvi_delta": random.uniform(-0.1, 0.1)}
            elif etype == "news_update":
                payload = {"env_news_delta": random.uniform(-0.2, 0.2)}
            else:
                payload = {"governance_flag": random.choice([0, 1])}
            prod.send(cid, etype, payload)
            time.sleep(1.0 / max(event_rate_hz, 1e-3))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
