use crate::infra::cache::CandidatePool;

#[test]
fn prev_on_single_candidate_does_not_panic() {
    // Regression: old `Some(self.cursor - 1)` underflowed here (0 - 1).
    let mut pool = CandidatePool::new();
    pool.push("a".into());
    assert_eq!(pool.prev(), Some(1));
    assert_eq!(pool.current(), Some(("a", 1, 1)));
}

#[test]
fn prev_from_oldest_wraps_without_underflow() {
    // Regression: wrapping onto index 0 used to panic / return garbage.
    let mut pool = CandidatePool::new();
    pool.push("a".into());
    pool.push("b".into());
    assert_eq!(pool.prev(), Some(1)); // v2 -> v1
    assert_eq!(pool.prev(), Some(2)); // v1 -> wraps to v2 (old code panicked)
    assert_eq!(pool.current(), Some(("b", 2, 2)));
}

#[test]
fn full_pool_evicts_oldest_and_total_stays_bounded() {
    let mut pool = CandidatePool::with_capacity(3);
    for m in ["a", "b", "c", "d"] {
        pool.push(m.to_string());
    }
    assert_eq!(pool.len(), 3); // "a" evicted; total derives from len()
    assert_eq!(pool.current(), Some(("d", 3, 3)));
    pool.prev();
    assert_eq!(pool.current(), Some(("c", 2, 3)));
    pool.prev();
    assert_eq!(pool.current(), Some(("b", 1, 3)));
    pool.prev();
    assert_eq!(pool.current(), Some(("d", 3, 3))); // wrap still sound
}

#[test]
fn duplicate_in_full_pool_reuses_slot_without_eviction() {
    let mut pool = CandidatePool::with_capacity(3);
    for m in ["a", "b", "c", "b"] {
        pool.push(m.to_string());
    }
    assert_eq!(pool.len(), 3);
    assert_eq!(pool.current(), Some(("b", 2, 3)));
}

#[test]
#[should_panic(expected = "non-zero")]
fn zero_capacity_is_rejected() {
    CandidatePool::with_capacity(0);
}
