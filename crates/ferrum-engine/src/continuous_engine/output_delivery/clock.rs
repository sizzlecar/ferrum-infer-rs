//! Fixed-size history of successful engine output-channel text enqueues.
//! Protocol filtering, transport coalescing and client receipt occur later.

use ferrum_scheduler::cost_prediction::RequestClockFacts;
use std::time::{Duration, Instant};

pub(super) struct DeliveryTextClock {
    ingress: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    maximum: Option<Duration>,
    events: u64,
    violating: u64,
    threshold: Option<Duration>,
    complete: bool,
}

impl DeliveryTextClock {
    pub(super) fn new(ingress: Instant, threshold: Option<Duration>) -> Self {
        Self {
            ingress,
            first: None,
            last: None,
            maximum: None,
            events: 0,
            violating: 0,
            threshold,
            complete: true,
        }
    }

    pub(super) fn record(&mut self, now: Instant) {
        if now < self.ingress || self.last.is_some_and(|last| now < last) {
            self.complete = false;
            return;
        }
        let Some(events) = self.events.checked_add(1) else {
            self.complete = false;
            return;
        };
        if let Some(last) = self.last {
            let interval = now - last;
            self.maximum = Some(self.maximum.unwrap_or_default().max(interval));
            if self.threshold.is_some_and(|threshold| interval > threshold) {
                let Some(violating) = self.violating.checked_add(1) else {
                    self.complete = false;
                    return;
                };
                self.violating = violating;
            }
        }
        self.first.get_or_insert(now);
        self.last = Some(now);
        self.events = events;
    }

    pub(super) fn snapshot(
        &self,
        now: Instant,
        generated_tokens: usize,
    ) -> Option<RequestClockFacts> {
        if !self.complete || now < self.ingress || self.last.is_some_and(|last| last > now) {
            return None;
        }
        Some(RequestClockFacts {
            captured_at: now,
            age: now - self.ingress,
            first_text: self.first.map(|first| first - self.ingress),
            last_text: self.last.map(|last| last - self.ingress),
            max_text_interval: self.maximum,
            nonempty_text_events: self.events,
            violating_text_intervals: self.violating,
            itl_threshold: self.threshold,
            generated_tokens,
            streaming: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_intervals_keep_historical_violations_and_original_ingress() {
        let ingress = Instant::now();
        let mut clock = DeliveryTextClock::new(ingress, Some(Duration::from_millis(5)));
        for offset in [10, 12, 20, 25] {
            clock.record(ingress + Duration::from_millis(offset));
        }
        let facts = clock
            .snapshot(ingress + Duration::from_millis(30), 9)
            .unwrap();
        assert_eq!(facts.age, Duration::from_millis(30));
        assert_eq!(facts.first_text, Some(Duration::from_millis(10)));
        assert_eq!(facts.last_text, Some(Duration::from_millis(25)));
        assert_eq!(facts.max_text_interval, Some(Duration::from_millis(8)));
        assert_eq!(facts.nonempty_text_events, 4);
        assert_eq!(facts.violating_text_intervals, 1);
        assert_eq!(facts.generated_tokens, 9);
    }

    #[test]
    fn missing_threshold_and_invalid_time_do_not_fabricate_measurements() {
        let now = Instant::now();
        let mut clock = DeliveryTextClock::new(now, None);
        assert_eq!(clock.snapshot(now, 0).unwrap().first_text, None);
        clock.record(now + Duration::from_millis(1));
        clock.record(now + Duration::from_secs(5));
        let facts = clock.snapshot(now + Duration::from_secs(5), 2).unwrap();
        assert_eq!(facts.itl_threshold, None);
        assert_eq!(facts.violating_text_intervals, 0);
        assert_eq!(facts.max_text_interval, Some(Duration::from_millis(4999)));
        clock.record(now);
        assert!(clock.snapshot(now + Duration::from_secs(6), 2).is_none());
    }
}
