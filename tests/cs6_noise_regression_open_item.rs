//! No bound assertion here: served-slot output-error distributions were measured
//! at gamma 16 and 256, but nothing in this file may be read as a symbolic bound.
//!
//! An assertion needs a bound on the served slot. `get_variance` in `src/params.rs` is
//! Theorem 7's pre-mod-switch variance (eprint 2025/1352): its two summands are the fold
//! term and the InspiRING packing term, and the `(q_tilde / q)^2` factor was reviewed and
//! does not belong in it. The additive mod-switch rounding term is charged only by
//! `check_mod_switch_noise_budget`, at its worst case. No single bound combines the two
//! for the served response. Byte-identity round-trips are not a substitute: they fail only
//! once noise exceeds Delta/2 and flips a byte, so they are blind to margin narrowing.
//!
//! Closing this needs all three, in order:
//!
//! 1. Derive one bound on the served slot's output error, the `get_variance` terms plus
//!    the mod-switch rounding term, so a bound exists to assert against.
//! 2. Extend the gamma 16/256 production measurements to every legal cell shape.
//! 3. Write a test demonstrated to fail against an injected noise regression.
//!    One that passes before and after the injection closes nothing.
//!
//! Until item 1 lands, the margin `get_variance` reports is an estimate, not a bound:
//! do not quote it as slack.
