//! Utilities.

pub mod arithmetic;
pub mod hash;
pub mod msm;
pub mod poly;
pub mod transcript;

pub(crate) use itertools::{chain, izip, Itertools};
pub(crate) use num_bigint::BigUint;
pub(crate) use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};
//pub(crate) use timer::{end_timer, start_timer, start_unit_timer};

macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::util::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        $crate::util::izip_eq!($first).zip_eq($second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        $crate::util::izip_eq!($first)
            $(.zip_eq($rest))*
            .map($crate::util::izip_eq!(@closure a => (a) $(, $rest)*))
    };
}

pub trait BitIndex {
    fn nth_bit(&self, nth: usize) -> bool;
}

impl BitIndex for usize {
    fn nth_bit(&self, nth: usize) -> bool {
        (self >> nth) & 1 == 1
    }
}

macro_rules! impl_index {
    (@ $name:ty, $field:tt, [$($range:ty => $output:ty),*$(,)?]) => {
        $(
            impl<F> std::ops::Index<$range> for $name {
                type Output = $output;

                fn index(&self, index: $range) -> &$output {
                    self.$field.index(index)
                }
            }

            impl<F> std::ops::IndexMut<$range> for $name {
                fn index_mut(&mut self, index: $range) -> &mut $output {
                    self.$field.index_mut(index)
                }
            }
        )*
    };
    (@ $name:ty, $field:tt) => {
        impl_index!(
            @ $name, $field,
            [
                usize => F,
                std::ops::Range<usize> => [F],
                std::ops::RangeFrom<usize> => [F],
                std::ops::RangeFull => [F],
                std::ops::RangeInclusive<usize> => [F],
                std::ops::RangeTo<usize> => [F],
                std::ops::RangeToInclusive<usize> => [F],
            ]
        );
    };
    ($name:ident, $field:tt) => {
        impl_index!(@ $name<F>, $field);
    };
}

pub(crate) use {impl_index, izip_eq};

#[cfg(feature = "parallel")]
pub(crate) use rayon::current_num_threads;

pub fn num_threads() -> usize {
    #[cfg(feature = "parallel")]
    return rayon::current_num_threads();

    #[cfg(not(feature = "parallel"))]
    return 1;
}

/// Parallelly executing the function on the items of the given iterator.
pub fn parallelize_iter<I, T, F>(iter: I, f: F)
where
    I: Send + Iterator<Item = T>,
    T: Send,
    F: Fn(T) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    rayon::scope(|scope| {
        for item in iter {
            let f = f.clone();
            scope.spawn(move |_| f(item));
        }
    });
    #[cfg(not(feature = "parallel"))]
    iter.for_each(f);
}

/// Parallelly executing the function on the given mutable slice.
pub fn parallelize<T, F>(v: &mut [T], f: F)
where
    T: Send,
    F: Fn((&mut [T], usize)) + Send + Sync + Clone,
{
    #[cfg(feature = "parallel")]
    {
        let num_threads = num_threads();
        let chunk_size = v.len() / num_threads;
        if chunk_size < num_threads {
            f((v, 0));
        } else {
            parallelize_iter(v.chunks_mut(chunk_size).zip((0..).step_by(chunk_size)), f);
        }
    }
    #[cfg(not(feature = "parallel"))]
    f((v, 0));
}

#[cfg(feature = "parallel")]
pub fn par_map_collect<T, R, C>(
    v: impl rayon::prelude::IntoParallelIterator<Item = T>,
    f: impl Fn(T) -> R + Send + Sync,
) -> C
where
    T: Send + Sync,
    R: Send,
    C: rayon::prelude::FromParallelIterator<R>,
{
    use rayon::prelude::ParallelIterator;
    v.into_par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
pub fn par_map_collect<T, R, C>(v: impl IntoIterator<Item = T>, f: impl Fn(T) -> R) -> C
where
    C: FromIterator<R>,
{
    v.into_iter().map(f).collect()
}

#[cfg(any(test, feature = "benchmark"))]
pub mod test {
    use crate::util::arithmetic::Field;
    use rand::{
        rngs::{OsRng, StdRng},
        CryptoRng, RngCore, SeedableRng,
    };
    use std::{array, iter, ops::Range};

    pub fn std_rng() -> impl RngCore + CryptoRng {
        StdRng::from_seed(Default::default())
    }

    pub fn seeded_std_rng() -> impl RngCore + CryptoRng {
        StdRng::seed_from_u64(OsRng.next_u64())
    }

    pub fn rand_idx(range: Range<usize>, mut rng: impl RngCore) -> usize {
        range.start + (rng.next_u64() as usize % (range.end - range.start))
    }

    pub fn rand_array<F: Field, const N: usize>(mut rng: impl RngCore) -> [F; N] {
        array::from_fn(|_| F::random(&mut rng))
    }

    pub fn rand_vec<F: Field>(n: usize, mut rng: impl RngCore) -> Vec<F> {
        iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
    }
}