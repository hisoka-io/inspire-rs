//! Arithmetic over R_q = Z_q\[X\]/(X^d + 1), negacyclic.
//!
//! ```
//! use raven_inspire::math::{Poly, NttContext};
//!
//! let ctx = NttContext::with_default_q(256);
//! let mut poly = Poly::random(256, ctx.modulus());
//! poly.to_ntt(&ctx);
//! ```

pub mod crt;
pub mod gaussian;
// Cross-check reference for the Solinas KAT suite; Solinas dispatch supersedes it for DEFAULT_Q.
pub mod ifma52;
pub mod mod_q;
pub mod modular;
pub mod ntt;
pub mod poly;
pub(crate) mod primality;
pub mod sampler;
pub mod sampling;
pub mod solinas_redc;

pub use crt::{crt_compose_2, crt_decompose_2, crt_modulus, mod_inverse};
pub use gaussian::GaussianSampler;
pub use mod_q::DEFAULT_Q;
pub use modular::ModQ;
pub use ntt::NttContext;
pub use poly::Poly;
