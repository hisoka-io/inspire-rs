//! No bound assertion here: W5-01 measured served-slot output-error distributions
//! at gamma 16 and 256, but nothing in this file may be read as a symbolic bound.
//!
//! An assertion needs a bound, and the in-crate packing-noise bound has an
//! unresolved `(q_tilde / q)^2` factor against eprint 2025/1352 Theorem 7.
//! Byte-identity round-trips are not a substitute: they fail only once noise
//! exceeds Delta/2 and flips a byte, so they are blind to margin narrowing.
//!
//! Closing this needs all three, in order:
//!
//! 1. Resolve the `(q_tilde / q)^2` factor against Theorem 7, so a bound exists
//!    to assert against.
//! 2. Extend the gamma 16/256 production measurements to every legal cell shape.
//! 3. Write a test demonstrated to fail against an injected noise regression.
//!    One that passes before and after the injection closes nothing.
//!
//! `get_variance` in `src/params.rs` is the entry point, and it already records
//! that the formula omits the InspiRING packing term, so the reported margin is
//! a known-wrong reading rather than a thin-but-passing one. Do not quote it as
//! slack until item 1 lands.
