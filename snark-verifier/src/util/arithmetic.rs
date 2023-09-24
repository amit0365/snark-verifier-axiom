//! Arithmetic related re-exported traits and utilities.

use crate::halo2_curves;
use crate::util::Itertools;
pub use halo2_curves::{
    group::{
        ff::{BatchInvert, Field, PrimeField},
        prime::PrimeCurveAffine,
        Curve, Group, GroupEncoding,
    },
    pairing::MillerLoopResult,
    Coordinates, CurveAffine, CurveExt, FieldExt,
};
use num_bigint::BigUint;
use num_traits::One;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::Debug,
    iter, mem,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use num_integer::Integer;

/// [`halo2_curves::pairing::MultiMillerLoop`] with [`std::fmt::Debug`].
pub trait MultiMillerLoop: halo2_curves::pairing::MultiMillerLoop + Debug {}

impl<M: halo2_curves::pairing::MultiMillerLoop + Debug> MultiMillerLoop for M {}

/// Operations that could be done with field elements.
pub trait FieldOps:
    Sized
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
{
    /// Returns multiplicative inversion if any.
    fn invert(&self) -> Option<Self>;
}

/// Batch invert [`PrimeField`] elements and multiply all with given coefficient.
pub fn batch_invert_and_mul<F: PrimeField>(values: &mut [F], coeff: &F) {
    if values.is_empty() {
        return;
    }
    let products = values
        .iter()
        .scan(F::one(), |acc, value| {
            *acc *= value;
            Some(*acc)
        })
        .collect_vec();

    let mut all_product_inv = Option::<F>::from(products.last().unwrap().invert())
        .expect("Attempted to batch invert an array containing zero")
        * coeff;

    for (value, product) in
        values.iter_mut().rev().zip(products.into_iter().rev().skip(1).chain(Some(F::one())))
    {
        let mut inv = all_product_inv * product;
        mem::swap(value, &mut inv);
        all_product_inv *= inv;
    }
}

/// Batch invert [`PrimeField`] elements.
pub fn batch_invert<F: PrimeField>(values: &mut [F]) {
    batch_invert_and_mul(values, &F::one())
}

/// Root of unity of 2^k-sized multiplicative subgroup of [`PrimeField`] by
/// repeatedly squaring the root of unity of the largest multiplicative
/// subgroup.
///
/// # Panic
///
/// If given `k` is greater than [`PrimeField::S`].
pub fn root_of_unity<F: PrimeField>(k: usize) -> F {
    assert!(k <= F::S as usize);

    iter::successors(Some(F::root_of_unity()), |acc| Some(acc.square()))
        .take(F::S as usize - k + 1)
        .last()
        .unwrap()
}

/// Rotation on a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Rotation(pub i32);

impl Rotation {
    /// No rotation
    pub fn cur() -> Self {
        Rotation(0)
    }

    /// To previous element
    pub fn prev() -> Self {
        Rotation(-1)
    }

    /// To next element
    pub fn next() -> Self {
        Rotation(1)
    }

    pub fn distance(&self) -> usize {
        self.0.unsigned_abs() as usize
    }
}

impl From<i32> for Rotation {
    fn from(rotation: i32) -> Self {
        Self(rotation)
    }
}

/// 2-adicity multiplicative domain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Domain<F: PrimeField> {
    /// Log size of the domain.
    pub k: usize,
    /// Size of the domain.
    pub n: usize,
    /// Inverse of `n`.
    pub n_inv: F,
    /// Generator of the domain.
    pub gen: F,
    /// Inverse of `gen`.
    pub gen_inv: F,
}

impl<F: PrimeField> Domain<F> {
    /// Initialize a domain with specified generator.
    pub fn new(k: usize, gen: F) -> Self {
        let n = 1 << k;
        let n_inv = F::from(n as u64).invert().unwrap();
        let gen_inv = gen.invert().unwrap();

        Self { k, n, n_inv, gen, gen_inv }
    }

    /// Rotate an element to given `rotation`.
    pub fn rotate_scalar(&self, scalar: F, rotation: Rotation) -> F {
        match rotation.0.cmp(&0) {
            Ordering::Equal => scalar,
            Ordering::Greater => scalar * self.gen.pow_vartime([rotation.0 as u64]),
            Ordering::Less => scalar * self.gen_inv.pow_vartime([(-(rotation.0 as i64)) as u64]),
        }
    }
}

/// Contains numerator and denominator for deferred evaluation.
#[derive(Clone, Debug)]
pub struct Fraction<T> {
    numer: Option<T>,
    denom: T,
    eval: Option<T>,
    inv: bool,
}

impl<T> Fraction<T> {
    /// Initialize an unevaluated fraction.
    pub fn new(numer: T, denom: T) -> Self {
        Self { numer: Some(numer), denom, eval: None, inv: false }
    }

    /// Initialize an unevaluated fraction without numerator.
    pub fn one_over(denom: T) -> Self {
        Self { numer: None, denom, eval: None, inv: false }
    }

    /// Returns denominator.
    pub fn denom(&self) -> Option<&T> {
        if !self.inv {
            Some(&self.denom)
        } else {
            None
        }
    }

    #[must_use = "To be inverted"]
    /// Returns mutable denominator for doing inversion.
    pub fn denom_mut(&mut self) -> Option<&mut T> {
        if !self.inv {
            self.inv = true;
            Some(&mut self.denom)
        } else {
            None
        }
    }
}

impl<T: FieldOps + Clone> Fraction<T> {
    /// Evaluate the fraction and cache the result.
    ///
    /// # Panic
    ///
    /// If `denom_mut` is not called before.
    pub fn evaluate(&mut self) {
        assert!(self.inv);

        if self.eval.is_none() {
            self.eval = Some(
                self.numer
                    .take()
                    .map(|numer| numer * &self.denom)
                    .unwrap_or_else(|| self.denom.clone()),
            );
        }
    }

    /// Returns cached fraction evaluation.
    ///
    /// # Panic
    ///
    /// If `evaluate` is not called before.
    pub fn evaluated(&self) -> &T {
        assert!(self.eval.is_some());

        self.eval.as_ref().unwrap()
    }
}

/// Modulus of a [`PrimeField`]
pub fn modulus<F: PrimeField>() -> BigUint {
    fe_to_big(-F::one()) + 1usize
}

/// Convert a [`BigUint`] into a [`PrimeField`] .
pub fn fe_from_big<F: PrimeField>(big: BigUint) -> F {
    let bytes = big.to_bytes_le();
    let mut repr = F::Repr::default();
    assert!(bytes.len() <= repr.as_ref().len());
    repr.as_mut()[..bytes.len()].clone_from_slice(bytes.as_slice());
    F::from_repr(repr).unwrap()
}

/// Convert a [`PrimeField`] into a [`BigUint`].
pub fn fe_to_big<F: PrimeField>(fe: F) -> BigUint {
    BigUint::from_bytes_le(fe.to_repr().as_ref())
}

/// Convert a [`PrimeField`] into another [`PrimeField`].
pub fn fe_to_fe<F1: PrimeField, F2: PrimeField>(fe: F1) -> F2 {
    fe_from_big(fe_to_big(fe) % modulus::<F2>())
}

/// Convert `LIMBS` limbs into a [`PrimeField`], assuming each limb contains at
/// most `BITS`.
pub fn fe_from_limbs<F1: PrimeField, F2: PrimeField, const LIMBS: usize, const BITS: usize>(
    limbs: [F1; LIMBS],
) -> F2 {
    fe_from_big(
        limbs
            .iter()
            .map(|limb| BigUint::from_bytes_le(limb.to_repr().as_ref()))
            .zip((0usize..).step_by(BITS))
            .map(|(limb, shift)| limb << shift)
            .reduce(|acc, shifted| acc + shifted)
            .unwrap(),
    )
}

/// Convert a [`PrimeField`] into `LIMBS` limbs where each limb contains at
/// most `BITS`.
pub fn fe_to_limbs<F1: PrimeField, F2: PrimeField, const LIMBS: usize, const BITS: usize>(
    fe: F1,
) -> [F2; LIMBS] {
    let big = BigUint::from_bytes_le(fe.to_repr().as_ref());
    let mask = &((BigUint::one() << BITS) - 1usize);
    (0usize..)
        .step_by(BITS)
        .take(LIMBS)
        .map(|shift| fe_from_big((&big >> shift) & mask))
        .collect_vec()
        .try_into()
        .unwrap()
}

/// Returns iterator that yields scalar^0, scalar^1, scalar^2...
pub fn powers<F: Field>(scalar: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::one()), move |power| Some(scalar * power))
}

/// Compute inner product of 2 slice of [`Field`].
pub fn inner_product<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    lhs.iter()
        .zip_eq(rhs.iter())
        .map(|(lhs, rhs)| *lhs * rhs)
        .reduce(|acc, product| acc + product)
        .unwrap_or_default()
}

use crate::util::par_map_collect;

/// Integer representation of primitive polynomial in GF(2).
const PRIMITIVES: [usize; 32] = [
    1,          // [0]
    3,          // [0, 1]
    7,          // [0, 1, 2]
    11,         // [0, 1, 3]
    19,         // [0, 1, 4]
    37,         // [0, 2, 5]
    67,         // [0, 1, 6]
    131,        // [0, 1, 7]
    285,        // [0, 2, 3, 4, 8]
    529,        // [0, 4, 9]
    1033,       // [0, 3, 10]
    2053,       // [0, 2, 11]
    4179,       // [0, 1, 4, 6, 12]
    8219,       // [0, 1, 3, 4, 13]
    16427,      // [0, 1, 3, 5, 14]
    32771,      // [0, 1, 15]
    65581,      // [0, 2, 3, 5, 16]
    131081,     // [0, 3, 17]
    262183,     // [0, 1, 2, 5, 18]
    524327,     // [0, 1, 2, 5, 19]
    1048585,    // [0, 3, 20]
    2097157,    // [0, 2, 21]
    4194307,    // [0, 1, 22]
    8388641,    // [0, 5, 23]
    16777243,   // [0, 1, 3, 4, 24]
    33554441,   // [0, 3, 25]
    67108935,   // [0, 1, 2, 6, 26]
    134217767,  // [0, 1, 2, 5, 27]
    268435465,  // [0, 3, 28]
    536870917,  // [0, 2, 29]
    1073741907, // [0, 1, 4, 6, 30]
    2147483657, // [0, 3, 31]
];

/// Integer representation of 1/X in GF(2).
const X_INVS: [usize; 32] = [
    0,          // []
    1,          // [0]
    3,          // [0, 1]
    5,          // [0, 2]
    9,          // [0, 3]
    18,         // [1, 4]
    33,         // [0, 5]
    65,         // [0, 6]
    142,        // [1, 2, 3, 7]
    264,        // [3, 8]
    516,        // [2, 9]
    1026,       // [1, 10]
    2089,       // [0, 3, 5, 11]
    4109,       // [0, 2, 3, 12]
    8213,       // [0, 2, 4, 13]
    16385,      // [0, 14]
    32790,      // [1, 2, 4, 15]
    65540,      // [2, 16]
    131091,     // [0, 1, 4, 17]
    262163,     // [0, 1, 4, 18]
    524292,     // [2, 19]
    1048578,    // [1, 20]
    2097153,    // [0, 21]
    4194320,    // [4, 22]
    8388621,    // [0, 2, 3, 23]
    16777220,   // [2, 24]
    33554467,   // [0, 1, 5, 25]
    67108883,   // [0, 1, 4, 26]
    134217732,  // [2, 27]
    268435458,  // [1, 28]
    536870953,  // [0, 3, 5, 29]
    1073741828, // [2, 30]
];

#[derive(Debug, Clone, Copy)]
pub struct BooleanHypercube {
    num_vars: usize,
    primitive: usize,
    x_inv: usize,
}

impl BooleanHypercube {
    pub const fn new(num_vars: usize) -> Self {
        assert!(num_vars < 32);
        Self {
            num_vars,
            primitive: PRIMITIVES[num_vars],
            x_inv: X_INVS[num_vars],
        }
    }

    pub const fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub const fn primitive(&self) -> usize {
        self.primitive
    }

    pub const fn x_inv(&self) -> usize {
        self.x_inv
    }

    pub fn rotate(&self, mut b: usize, Rotation(rotation): Rotation) -> usize {
        match rotation.cmp(&0) {
            Ordering::Equal => {}
            Ordering::Less => {
                for _ in rotation..0 {
                    b = prev(b, self.x_inv);
                }
            }
            Ordering::Greater => {
                for _ in 0..rotation {
                    b = next(b, self.num_vars, self.primitive);
                }
            }
        };
        b
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        iter::once(0)
            .chain(iter::successors(Some(1), |b| {
                next(*b, self.num_vars, self.primitive).into()
            }))
            .take(1 << self.num_vars)
    }

    pub fn nth_map(&self) -> Vec<usize> {
        let mut nth_map = vec![0; 1 << self.num_vars];
        for (nth, b) in self.iter().enumerate() {
            nth_map[b] = nth;
        }
        nth_map
    }

    pub fn rotation_map(&self, rotation: Rotation) -> Vec<usize> {
        par_map_collect(0..1 << self.num_vars, |b| self.rotate(b, rotation))
    }
}

#[inline(always)]
fn next(mut b: usize, num_vars: usize, primitive: usize) -> usize {
    b <<= 1;
    b ^= (b >> num_vars) * primitive;
    b
}

#[inline(always)]
fn prev(b: usize, x_inv: usize) -> usize {
    (b >> 1) ^ ((b & 1) * x_inv)
}

#[cfg(test)]
mod test {
    use crate::util::arithmetic::{BooleanHypercube,Rotation};

    #[test]
    #[ignore = "cause it takes some minutes to run with release profile"]
    fn boolean_hypercube_iter() {
        for num_vars in 0..32 {
            let bh = BooleanHypercube::new(num_vars);
            let mut set = vec![false; 1 << num_vars];
            for i in bh.iter() {
                assert!(!set[i]);
                set[i] = true;
            }
        }
    }

    #[test]
    #[ignore = "cause it takes some minutes to run with release profile"]
    fn boolean_hypercube_prev() {
        for num_vars in 0..32 {
            let bh = BooleanHypercube::new(num_vars);
            for (b, b_next) in bh.iter().skip(1).zip(bh.iter().skip(2).chain(Some(1))) {
                assert_eq!(b, bh.rotate(b_next, Rotation::prev()))
            }
        }
    }
}

pub fn usize_from_bits_le(bits: &[bool]) -> usize {
    bits.iter()
        .rev()
        .fold(0, |int, bit| (int << 1) + (*bit as usize))
}

pub fn div_rem(dividend: usize, divisor: usize) -> (usize, usize) {
    Integer::div_rem(&dividend, &divisor)
}

pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
    Integer::div_ceil(&dividend, &divisor)
}

pub fn horner<F: Field>(coeffs: &[F], x: &F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::zero(), |acc, coeff| acc * x + coeff)
}