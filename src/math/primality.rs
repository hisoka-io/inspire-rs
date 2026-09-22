//! Exact primality for the public moduli an NTT context is built on.

/// The first twelve primes. The least composite that is a strong pseudoprime to all of them
/// is `318_665_857_834_031_151_167_461 > 2^64` (Sorenson and Webster, arXiv:1509.00864,
/// Theorem 1.1), so Miller-Rabin over these bases is exact on `u64`.
const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(modulus)) as u64
}

fn pow_mod(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut power = 1;
    base %= modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            power = mul_mod(power, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exponent >>= 1;
    }
    power
}

/// Whether `modulus` is prime. Variable-time in `modulus`, which is a public parameter; it
/// runs when a parameter set is validated or a context is built, never per coefficient.
pub(crate) fn is_prime(modulus: u64) -> bool {
    if modulus < 2 {
        return false;
    }
    for witness in WITNESSES {
        if modulus.is_multiple_of(witness) {
            return modulus == witness;
        }
    }
    let twos = (modulus - 1).trailing_zeros();
    let odd = (modulus - 1) >> twos;
    WITNESSES.into_iter().all(|witness| {
        let mut power = pow_mod(witness, odd, modulus);
        if power == 1 || power == modulus - 1 {
            return true;
        }
        for _ in 1..twos {
            power = mul_mod(power, power, modulus);
            if power == modulus - 1 {
                return true;
            }
        }
        false
    })
}

#[cfg(test)]
mod tests {
    use super::is_prime;
    use crate::math::mod_q::DEFAULT_Q;
    use crate::params::{DEFAULT_CRT_MODULI, DEFAULT_Q_2CRT_30BIT};

    fn is_prime_by_trial_division(n: u64) -> bool {
        n >= 2 && (2..=n.isqrt()).all(|f| !n.is_multiple_of(f))
    }

    #[test]
    fn agrees_with_trial_division_below_two_to_the_sixteenth() {
        for n in 0..=u64::from(u16::MAX) {
            assert_eq!(is_prime(n), is_prime_by_trial_division(n), "{n}");
        }
    }

    #[test]
    fn every_modulus_the_crate_ships_is_prime() {
        let mut shipped = vec![DEFAULT_Q, 65_537, 67_043_329, 132_120_577];
        shipped.extend(DEFAULT_CRT_MODULI);
        shipped.extend(DEFAULT_Q_2CRT_30BIT);
        #[cfg(feature = "mod-switch-response")]
        shipped.extend(crate::pir::mod_switch::IMPLEMENTED_TARGETS);
        for modulus in shipped {
            assert!(is_prime(modulus), "{modulus}");
        }
    }

    #[test]
    fn composites_built_to_pass_weaker_tests_are_refused() {
        // Carmichael numbers: Fermat liars to every coprime base.
        for carmichael in [561, 1105, 1729, 294_409, 56_052_361, 2_718_557_844_481] {
            assert!(!is_prime(carmichael), "{carmichael}");
        }
        // Strong pseudoprimes to base 2, to bases 2..=7, and to the first eleven primes:
        // only the twelfth witness, 37, refuses the last.
        for strong_pseudoprime in [2047, 3_215_031_751, 3_825_123_056_546_413_051] {
            assert!(!is_prime(strong_pseudoprime), "{strong_pseudoprime}");
        }
        // NTT-friendly composites a server has actually been tested with.
        for forged in [68_719_309_313, 68_718_424_065, 1_073_725_441, 1_072_693_249] {
            assert!(!is_prime(forged), "{forged}");
        }
    }

    #[test]
    fn the_edges_of_the_range_are_exact() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(37));
        assert!(!is_prime(37 * 37));
        assert!(!is_prime(37 * 41));
        // 2^64 - 59, the largest prime below 2^64, and its composite neighbours.
        assert!(is_prime(u64::MAX - 58));
        assert!(!is_prime(u64::MAX));
        assert!(!is_prime(u64::MAX - 2));
    }
}
