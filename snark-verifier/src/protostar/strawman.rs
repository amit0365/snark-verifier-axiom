mod strawman {
    use crate::{
        accumulation::protostar::{
            ivc::halo2::{
                AssignedProtostarAccumulatorInstance, HashInstruction, TranscriptInstruction,
                TwoChainCurveInstruction,
            },
            ProtostarAccumulatorInstance,
        },
        frontend::halo2::chip::halo2_wrong::{
            from_le_bits, integer_to_native, sum_with_coeff, to_le_bits_strict, PoseidonChip,
        },
        util::{
            arithmetic::{
                fe_from_bool, fe_from_le_bytes, fe_to_fe, fe_truncated, BitField, CurveAffine,
                Field, FromUniformBytes, Group, PrimeCurveAffine, PrimeField, PrimeFieldBits,
                TwoChainCurve,
            },
            hash::{poseidon::Spec, Poseidon},
            izip_eq,
            transcript::{
                FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript,
                Transcript, TranscriptRead, TranscriptWrite,
            },
            Itertools,
        },
    };
    use halo2_proofs::{
        circuit::{AssignedCell, Layouter, Value},
        plonk::{Column, ConstraintSystem, Error, Instance},
    };
    use halo2_wrong_v2::{
        integer::{
            chip::{IntegerChip, Range},
            rns::Rns,
            Integer,
        },
        maingate::{config::MainGate, operations::Collector, Gate},
        Composable, Scaled, Witness,
    };
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        fmt::{self, Debug},
        io::{self, Cursor, Read},
        iter,
        marker::PhantomData,
        ops::DerefMut,
        rc::Rc,
    };

    pub const NUM_LIMBS: usize = 4;
    pub const NUM_LIMB_BITS: usize = 65;
    const NUM_SUBLIMBS: usize = 5;
    const NUM_LOOKUPS: usize = 1;

    const T: usize = 5;
    const RATE: usize = 4;
    const R_F: usize = 8;
    const R_P: usize = 60;

    const NUM_CHALLENGE_BITS: usize = 128;
    const NUM_CHALLENGE_BYTES: usize = NUM_CHALLENGE_BITS / 8;

    const NUM_HASH_BITS: usize = 250;

    fn fe_to_limbs<F1: PrimeFieldBits, F2: PrimeField>(fe: F1, num_limb_bits: usize) -> Vec<F2> {
        fe.to_le_bits()
            .chunks(num_limb_bits)
            .into_iter()
            .map(|bits| match bits.len() {
                1..=64 => F2::from(bits.load_le()),
                65..=128 => {
                    let lo = bits[..64].load_le::<u64>();
                    let hi = bits[64..].load_le::<u64>();
                    F2::from(hi) * F2::from(2).pow_vartime([64]) + F2::from(lo)
                }
                _ => unimplemented!(),
            })
            .take(NUM_LIMBS)
            .collect()
    }

    pub fn fe_from_limbs<F1: PrimeFieldBits, F2: PrimeField>(
        limbs: &[F1],
        num_limb_bits: usize,
    ) -> F2 {
        limbs.iter().rev().fold(F2::ZERO, |acc, limb| {
            acc * F2::from_u128(1 << num_limb_bits) + fe_to_fe::<F1, F2>(*limb)
        })
    }

    fn x_y_is_identity<C: CurveAffine>(ec_point: &C) -> [C::Base; 3] {
        let coords = ec_point.coordinates().unwrap();
        let is_identity = (coords.x().is_zero() & coords.y().is_zero()).into();
        [*coords.x(), *coords.y(), fe_from_bool(is_identity)]
    }

    pub fn accumulation_transcript_param<F: FromUniformBytes<64>>() -> Spec<F, T, RATE> {
        Spec::new(R_F, R_P)
    }

    pub fn decider_transcript_param<F: FromUniformBytes<64>>() -> Spec<F, T, RATE> {
        Spec::new(R_F, R_P)
    }

    pub trait TwoChainCurveInstruction<C: TwoChainCurve>: Clone + Debug {
        type Config: Clone + Debug;
        type Assigned: Clone + Debug;
        type AssignedBase: Clone + Debug;
        type AssignedPrimary: Clone + Debug;
        type AssignedSecondary: Clone + Debug;
    
        fn new(config: Self::Config) -> Self;
    
        fn to_assigned(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            assigned: &AssignedCell<C::Scalar, C::Scalar>,
        ) -> Result<Self::Assigned, Error>;
    
        fn constrain_instance(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            value: &Self::Assigned,
            row: usize,
        ) -> Result<(), Error>;
    
        fn constrain_equal(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<(), Error>;
    
        fn assign_constant(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            constant: C::Scalar,
        ) -> Result<Self::Assigned, Error>;
    
        fn assign_witness(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Scalar>,
        ) -> Result<Self::Assigned, Error>;
    
        fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C::Scalar) -> bool);
    
        fn select(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error>;
    
        fn is_equal(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error>;
    
        fn add(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error>;
    
        fn sub(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error>;
    
        fn mul(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error>;
    
        fn constrain_equal_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<(), Error>;
    
        fn assign_constant_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            constant: C::Base,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn assign_witness_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Base>,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn assert_if_known_base(&self, value: &Self::AssignedBase, f: impl FnOnce(&C::Base) -> bool);
    
        fn select_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::AssignedBase,
            when_false: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn fit_base_in_scalar(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Self::Assigned, Error>;
    
        fn to_repr_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Vec<Self::Assigned>, Error>;
    
        fn add_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn sub_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn neg_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let zero = self.assign_constant_base(layouter, C::Base::ZERO)?;
            self.sub_base(layouter, &zero, value)
        }
    
        fn sum_base<'a>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            values: impl IntoIterator<Item = &'a Self::AssignedBase>,
        ) -> Result<Self::AssignedBase, Error>
        where
            Self::AssignedBase: 'a,
        {
            values.into_iter().fold(
                self.assign_constant_base(layouter, C::Base::ZERO),
                |acc, value| self.add_base(layouter, &acc?, value),
            )
        }
    
        fn product_base<'a>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            values: impl IntoIterator<Item = &'a Self::AssignedBase>,
        ) -> Result<Self::AssignedBase, Error>
        where
            Self::AssignedBase: 'a,
        {
            values.into_iter().fold(
                self.assign_constant_base(layouter, C::Base::ONE),
                |acc, value| self.mul_base(layouter, &acc?, value),
            )
        }
    
        fn mul_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn div_incomplete_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error>;
    
        fn invert_incomplete_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let one = self.assign_constant_base(layouter, C::Base::ONE)?;
            self.div_incomplete_base(layouter, &one, value)
        }
    
        fn powers_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            base: &Self::AssignedBase,
            n: usize,
        ) -> Result<Vec<Self::AssignedBase>, Error> {
            Ok(match n {
                0 => Vec::new(),
                1 => vec![self.assign_constant_base(layouter, C::Base::ONE)?],
                2 => vec![
                    self.assign_constant_base(layouter, C::Base::ONE)?,
                    base.clone(),
                ],
                _ => {
                    let mut powers = Vec::with_capacity(n);
                    powers.push(self.assign_constant_base(layouter, C::Base::ONE)?);
                    powers.push(base.clone());
                    for _ in 0..n - 2 {
                        powers.push(self.mul_base(layouter, powers.last().unwrap(), base)?);
                    }
                    powers
                }
            })
        }
    
        fn squares_base(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            base: &Self::AssignedBase,
            n: usize,
        ) -> Result<Vec<Self::AssignedBase>, Error> {
            Ok(match n {
                0 => Vec::new(),
                1 => vec![base.clone()],
                _ => {
                    let mut squares = Vec::with_capacity(n);
                    squares.push(base.clone());
                    for _ in 0..n - 1 {
                        squares.push(self.mul_base(
                            layouter,
                            squares.last().unwrap(),
                            squares.last().unwrap(),
                        )?);
                    }
                    squares
                }
            })
        }
    
        fn inner_product_base<'a, 'b>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: impl IntoIterator<Item = &'a Self::AssignedBase>,
            rhs: impl IntoIterator<Item = &'b Self::AssignedBase>,
        ) -> Result<Self::AssignedBase, Error>
        where
            Self::AssignedBase: 'a + 'b,
        {
            let products = izip_eq!(lhs, rhs)
                .map(|(lhs, rhs)| self.mul_base(layouter, lhs, rhs))
                .collect_vec();
            products
                .into_iter()
                .reduce(|acc, output| self.add_base(layouter, &acc?, &output?))
                .unwrap()
        }
    
        fn constrain_equal_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedSecondary,
            rhs: &Self::AssignedSecondary,
        ) -> Result<(), Error>;
    
        fn assign_constant_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            constant: C::Secondary,
        ) -> Result<Self::AssignedSecondary, Error>;
    
        fn assign_witness_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Secondary>,
        ) -> Result<Self::AssignedSecondary, Error>;
    
        fn assert_if_known_secondary(
            &self,
            value: &Self::AssignedSecondary,
            f: impl FnOnce(&C::Secondary) -> bool,
        );
    
        fn select_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::AssignedSecondary,
            when_false: &Self::AssignedSecondary,
        ) -> Result<Self::AssignedSecondary, Error>;
    
        fn add_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedSecondary,
            rhs: &Self::AssignedSecondary,
        ) -> Result<Self::AssignedSecondary, Error>;
    
        fn scalar_mul_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            base: &Self::AssignedSecondary,
            scalar_le_bits: &[Self::Assigned],
        ) -> Result<Self::AssignedSecondary, Error>;
    
        fn fixed_base_msm_secondary<'a, 'b>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            bases: impl IntoIterator<Item = &'a C::Secondary>,
            scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
        ) -> Result<Self::AssignedSecondary, Error>
        where
            Self::AssignedBase: 'b;
    
        fn variable_base_msm_secondary<'a, 'b>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            bases: impl IntoIterator<Item = &'a Self::AssignedSecondary>,
            scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
        ) -> Result<Self::AssignedSecondary, Error>
        where
            Self::AssignedSecondary: 'a,
            Self::AssignedBase: 'b;
    }

    
    #[derive(Debug)]
    pub struct PoseidonTranscript<F: PrimeField, S> {
        state: Poseidon<F, T, RATE>,
        stream: S,
    }

    impl<F: FromUniformBytes<64>> InMemoryTranscript for PoseidonTranscript<F, Cursor<Vec<u8>>> {
        type Param = Spec<F, T, RATE>;

        fn new(spec: Self::Param) -> Self {
            Self {
                state: Poseidon::new_with_spec(spec),
                stream: Default::default(),
            }
        }

        fn into_proof(self) -> Vec<u8> {
            self.stream.into_inner()
        }

        fn from_proof(spec: Self::Param, proof: &[u8]) -> Self {
            Self {
                state: Poseidon::new_with_spec(spec),
                stream: Cursor::new(proof.to_vec()),
            }
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, S> FieldTranscript<F>
        for PoseidonTranscript<N, S>
    {
        fn squeeze_challenge(&mut self) -> F {
            let hash = self.state.squeeze();
            self.state.update(&[hash]);

            fe_from_le_bytes(&hash.to_repr().as_ref()[..NUM_CHALLENGE_BYTES])
        }

        fn common_field_element(&mut self, fe: &F) -> Result<(), crate::Error> {
            self.state.update(&fe_to_limbs(*fe, NUM_LIMB_BITS));

            Ok(())
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, R: io::Read> FieldTranscriptRead<F>
        for PoseidonTranscript<N, R>
    {
        fn read_field_element(&mut self) -> Result<F, crate::Error> {
            let mut repr = <F as PrimeField>::Repr::default();
            self.stream
                .read_exact(repr.as_mut())
                .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            let fe = F::from_repr_vartime(repr).ok_or_else(|| {
                crate::Error::Transcript(
                    io::ErrorKind::Other,
                    "Invalid field element encoding in proof".to_string(),
                )
            })?;
            self.common_field_element(&fe)?;
            Ok(fe)
        }
    }

    impl<F: PrimeFieldBits, N: FromUniformBytes<64>, W: io::Write> FieldTranscriptWrite<F>
        for PoseidonTranscript<N, W>
    {
        fn write_field_element(&mut self, fe: &F) -> Result<(), crate::Error> {
            self.common_field_element(fe)?;
            let repr = fe.to_repr();
            self.stream
                .write_all(repr.as_ref())
                .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))
        }
    }

    impl<C: CurveAffine, S> Transcript<C, C::Scalar> for PoseidonTranscript<C::Base, S>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn common_commitment(&mut self, ec_point: &C) -> Result<(), crate::Error> {
            self.state.update(&x_y_is_identity(ec_point));
            Ok(())
        }
    }

    impl<C: CurveAffine, R: io::Read> TranscriptRead<C, C::Scalar> for PoseidonTranscript<C::Base, R>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn read_commitment(&mut self) -> Result<C, crate::Error> {
            let mut reprs = [<C::Base as PrimeField>::Repr::default(); 2];
            for repr in &mut reprs {
                self.stream
                    .read_exact(repr.as_mut())
                    .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            }
            let [x, y] = reprs.map(<C::Base as PrimeField>::from_repr_vartime);
            let ec_point = x
                .zip(y)
                .and_then(|(x, y)| CurveAffine::from_xy(x, y).into())
                .ok_or_else(|| {
                    crate::Error::Transcript(
                        io::ErrorKind::Other,
                        "Invalid elliptic curve point encoding in proof".to_string(),
                    )
                })?;
            self.common_commitment(&ec_point)?;
            Ok(ec_point)
        }
    }

    impl<C: CurveAffine, W: io::Write> TranscriptWrite<C, C::Scalar> for PoseidonTranscript<C::Base, W>
    where
        C::Base: FromUniformBytes<64>,
        C::Scalar: PrimeFieldBits,
    {
        fn write_commitment(&mut self, ec_point: &C) -> Result<(), crate::Error> {
            self.common_commitment(ec_point)?;
            let coordinates = ec_point.coordinates().unwrap();
            for coordinate in [coordinates.x(), coordinates.y()] {
                let repr = coordinate.to_repr();
                self.stream
                    .write_all(repr.as_ref())
                    .map_err(|err| crate::Error::Transcript(err.kind(), err.to_string()))?;
            }
            Ok(())
        }
    }

    #[derive(Clone, Debug)]
    pub struct Config<F: PrimeField> {
        pub main_gate: MainGate<F, NUM_LOOKUPS>,
        pub instance: Column<Instance>,
        pub poseidon_spec: Spec<F, T, RATE>,
    }

    impl<F: FromUniformBytes<64> + Ord> Config<F> {
        pub fn configure<C: CurveAffine<ScalarExt = F>>(meta: &mut ConstraintSystem<F>) -> Self {
            let rns =
                Rns::<C::Base, C::Scalar, NUM_LIMBS, NUM_LIMB_BITS, NUM_SUBLIMBS>::construct();
            let overflow_bit_lens = rns.overflow_lengths();
            let composition_bit_len = IntegerChip::<
                C::Base,
                C::Scalar,
                NUM_LIMBS,
                NUM_LIMB_BITS,
                NUM_SUBLIMBS,
            >::sublimb_bit_len();
            let main_gate = MainGate::<_, NUM_LOOKUPS>::configure(
                meta,
                vec![composition_bit_len],
                overflow_bit_lens,
            );
            let instance = meta.instance_column();
            meta.enable_equality(instance);
            let poseidon_spec = Spec::new(R_F, R_P);
            Self {
                main_gate,
                instance,
                poseidon_spec,
            }
        }
    }

    #[allow(clippy::type_complexity)]
    #[derive(Clone, Debug)]
    pub struct Chip<C: CurveAffine> {
        rns: Rns<C::Base, C::Scalar, NUM_LIMBS, NUM_LIMB_BITS, NUM_SUBLIMBS>,
        pub main_gate: MainGate<C::Scalar, NUM_LOOKUPS>,
        pub collector: Rc<RefCell<Collector<C::Scalar>>>,
        pub cell_map: Rc<RefCell<BTreeMap<u32, AssignedCell<C::Scalar, C::Scalar>>>>,
        pub instance: Column<Instance>,
        poseidon_spec: Spec<C::Scalar, T, RATE>,
        _marker: PhantomData<C>,
    }

    impl<C: TwoChainCurve> Chip<C> {
        #[allow(clippy::type_complexity)]
        pub fn layout_and_clear(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
        ) -> Result<BTreeMap<u32, AssignedCell<C::Scalar, C::Scalar>>, Error> {
            let cell_map = self.main_gate.layout(layouter, &self.collector.borrow())?;
            *self.collector.borrow_mut() = Default::default();
            Ok(cell_map)
        }

        fn double_ec_point_incomplete(
            &self,
            value: &AssignedEcPoint<C::Secondary>,
        ) -> AssignedEcPoint<C::Secondary> {
            let collector = &mut self.collector.borrow_mut();
            let two = C::Scalar::ONE.double();
            let three = two + C::Scalar::ONE;
            let lambda_numer =
                collector.mul_add_constant_scaled(three, value.x(), value.x(), C::Secondary::a());
            let y_doubled = collector.add(value.y(), value.y());
            let (lambda_denom_inv, _) = collector.inv(&y_doubled);
            let lambda = collector.mul(&lambda_numer, &lambda_denom_inv);
            let lambda_square = collector.mul(&lambda, &lambda);
            let out_x = collector.add_scaled(
                &Scaled::new(&lambda_square, C::Scalar::ONE),
                &Scaled::new(value.x(), -two),
            );
            let out_y = {
                let x_diff = collector.sub(value.x(), &out_x);
                let lambda_x_diff = collector.mul(&lambda, &x_diff);
                collector.sub(&lambda_x_diff, value.y())
            };
            AssignedEcPoint {
                ec_point: (value.ec_point + value.ec_point).map(Into::into),
                x: out_x,
                y: out_y,
                is_identity: *value.is_identity(),
            }
        }

        #[allow(clippy::type_complexity)]
        fn add_ec_point_inner(
            &self,
            lhs: &AssignedEcPoint<C::Secondary>,
            rhs: &AssignedEcPoint<C::Secondary>,
        ) -> (
            AssignedEcPoint<C::Secondary>,
            Witness<C::Scalar>,
            Witness<C::Scalar>,
        ) {
            let collector = &mut self.collector.borrow_mut();
            let x_diff = collector.sub(rhs.x(), lhs.x());
            let y_diff = collector.sub(rhs.y(), lhs.y());
            let (x_diff_inv, is_x_equal) = collector.inv(&x_diff);
            let (_, is_y_equal) = collector.inv(&y_diff);
            let lambda = collector.mul(&y_diff, &x_diff_inv);
            let lambda_square = collector.mul(&lambda, &lambda);
            let out_x = sum_with_coeff(
                collector,
                [
                    (&lambda_square, C::Scalar::ONE),
                    (lhs.x(), -C::Scalar::ONE),
                    (rhs.x(), -C::Scalar::ONE),
                ],
            );
            let out_y = {
                let x_diff = collector.sub(lhs.x(), &out_x);
                let lambda_x_diff = collector.mul(&lambda, &x_diff);
                collector.sub(&lambda_x_diff, lhs.y())
            };
            let out_x = collector.select(rhs.is_identity(), lhs.x(), &out_x);
            let out_x = collector.select(lhs.is_identity(), rhs.x(), &out_x);
            let out_y = collector.select(rhs.is_identity(), lhs.y(), &out_y);
            let out_y = collector.select(lhs.is_identity(), rhs.y(), &out_y);
            let out_is_identity = collector.mul(lhs.is_identity(), rhs.is_identity());

            let out = AssignedEcPoint {
                ec_point: (lhs.ec_point + rhs.ec_point).map(Into::into),
                x: out_x,
                y: out_y,
                is_identity: out_is_identity,
            };
            (out, is_x_equal, is_y_equal)
        }

        fn double_ec_point(
            &self,
            value: &AssignedEcPoint<C::Secondary>,
        ) -> AssignedEcPoint<C::Secondary> {
            let doubled = self.double_ec_point_incomplete(value);
            let collector = &mut self.collector.borrow_mut();
            let zero = collector.register_constant(C::Scalar::ZERO);
            let out_x = collector.select(value.is_identity(), &zero, doubled.x());
            let out_y = collector.select(value.is_identity(), &zero, doubled.y());
            AssignedEcPoint {
                ec_point: (value.ec_point + value.ec_point).map(Into::into),
                x: out_x,
                y: out_y,
                is_identity: *value.is_identity(),
            }
        }
    }

    #[derive(Clone)]
    pub struct AssignedBase<F: PrimeField, N: PrimeField> {
        scalar: Integer<F, N, NUM_LIMBS, NUM_LIMB_BITS>,
        limbs: Vec<Witness<N>>,
    }

    impl<F: PrimeField, N: PrimeField> AssignedBase<F, N> {
        fn assigned_cells(&self) -> impl Iterator<Item = &Witness<N>> {
            self.limbs.iter()
        }
    }

    impl<F: PrimeField, N: PrimeField> Debug for AssignedBase<F, N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedBase");
            let mut value = None;
            self.scalar.value().map(|scalar| value = Some(scalar));
            if let Some(value) = value {
                s.field("scalar", &value).finish()
            } else {
                s.finish()
            }
        }
    }

    #[derive(Clone)]
    pub struct AssignedEcPoint<C: CurveAffine> {
        ec_point: Value<C>,
        x: Witness<C::Base>,
        y: Witness<C::Base>,
        is_identity: Witness<C::Base>,
    }

    impl<C: CurveAffine> AssignedEcPoint<C> {
        pub fn x(&self) -> &Witness<C::Base> {
            &self.x
        }

        pub fn y(&self) -> &Witness<C::Base> {
            &self.y
        }

        pub fn is_identity(&self) -> &Witness<C::Base> {
            &self.is_identity
        }

        fn assigned_cells(&self) -> impl Iterator<Item = &Witness<C::Base>> {
            [self.x(), self.y(), self.is_identity()].into_iter()
        }
    }

    impl<C: CurveAffine> Debug for AssignedEcPoint<C> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut s = f.debug_struct("AssignedEcPoint");
            let mut value = None;
            self.ec_point.map(|ec_point| value = Some(ec_point));
            if let Some(value) = value {
                s.field("ec_point", &value).finish()
            } else {
                s.finish()
            }
        }
    }

    impl<C: TwoChainCurve> TwoChainCurveInstruction<C> for Chip<C> {
        type Config = Config<C::Scalar>;
        type Assigned = Witness<C::Scalar>;
        type AssignedBase = AssignedBase<C::Base, C::Scalar>;
        type AssignedPrimary = Vec<Witness<C::Scalar>>;
        type AssignedSecondary = AssignedEcPoint<C::Secondary>;

        fn new(config: Self::Config) -> Self {
            Chip {
                rns: Rns::construct(),
                main_gate: config.main_gate,
                collector: Default::default(),
                cell_map: Default::default(),
                instance: config.instance,
                poseidon_spec: config.poseidon_spec,
                _marker: PhantomData,
            }
        }

        fn to_assigned(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            value: &AssignedCell<C::Scalar, C::Scalar>,
        ) -> Result<Self::Assigned, Error> {
            Ok(self.collector.borrow_mut().new_external(value))
        }

        fn constrain_instance(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            assigned: &Self::Assigned,
            row: usize,
        ) -> Result<(), Error> {
            let cell = match row {
                0 => {
                    *self.cell_map.borrow_mut() =
                        self.main_gate.layout(layouter, &self.collector.borrow())?;
                    self.cell_map.borrow()[&assigned.id()].cell()
                }
                1 => {
                    *self.collector.borrow_mut() = Default::default();
                    let cell_map = std::mem::take(self.cell_map.borrow_mut().deref_mut());
                    cell_map[&assigned.id()].cell()
                }
                _ => unreachable!(),
            };

            layouter.constrain_instance(cell, self.instance, row)?;

            Ok(())
        }

        fn constrain_equal(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<(), Error> {
            lhs.value()
                .zip(rhs.value())
                .assert_if_known(|(lhs, rhs)| lhs == rhs);
            self.collector.borrow_mut().equal(lhs, rhs);
            Ok(())
        }
       
        fn assign_constant(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            constant: C::Scalar,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.register_constant(constant))
        }

        fn assign_witness(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Scalar>,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let value = collector.new_witness(witness);
            Ok(collector.add_constant(&value, C::Scalar::ZERO))
        }

        fn assert_if_known(&self, value: &Self::Assigned, f: impl FnOnce(&C::Scalar) -> bool) {
            value.value().assert_if_known(f)
        }

        fn select(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::Assigned,
            when_false: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.select(condition, when_true, when_false))
        }

        fn is_equal(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.is_equal(lhs, rhs))
        }

        fn add(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.add(lhs, rhs))
        }

        fn sub(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.sub(lhs, rhs))
        }

        fn mul(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::Assigned,
            rhs: &Self::Assigned,
        ) -> Result<Self::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            Ok(collector.mul(lhs, rhs))
        }

        fn constrain_equal_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<(), Error> {
            lhs.scalar
                .value()
                .zip(rhs.scalar.value())
                .assert_if_known(|(lhs, rhs)| lhs == rhs);
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            integer_chip.assert_equal(&lhs.scalar, &rhs.scalar);
            Ok(())
        }

        fn assign_constant_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            constant: C::Base,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.register_constant(constant);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn assign_witness_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Base>,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.range(self.rns.from_fe(witness), Range::Remainder);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn assert_if_known_base(
            &self,
            value: &Self::AssignedBase,
            f: impl FnOnce(&C::Base) -> bool,
        ) {
            value.scalar.value().assert_if_known(f)
        }

        fn select_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::AssignedBase,
            when_false: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.select(&when_true.scalar, &when_false.scalar, condition);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn fit_base_in_scalar(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Self::Assigned, Error> {
            Ok(integer_to_native(
                &self.rns,
                &mut self.collector.borrow_mut(),
                &value.scalar,
                NUM_HASH_BITS,
            ))
        }

        fn to_repr_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            value: &Self::AssignedBase,
        ) -> Result<Vec<Self::Assigned>, Error> {
            Ok(value.limbs.clone())
        }

        fn add_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.add(&lhs.scalar, &rhs.scalar);
            let scalar = integer_chip.reduce(&scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn sub_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.sub(&lhs.scalar, &rhs.scalar);
            let scalar = integer_chip.reduce(&scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn mul_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.mul(&lhs.scalar, &rhs.scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn div_incomplete_base(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedBase,
            rhs: &Self::AssignedBase,
        ) -> Result<Self::AssignedBase, Error> {
            let collector = &mut self.collector.borrow_mut();
            let mut integer_chip = IntegerChip::new(collector, &self.rns);
            let scalar = integer_chip.div_incomplete(&lhs.scalar, &rhs.scalar);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();
            Ok(AssignedBase { scalar, limbs })
        }

        fn constrain_equal_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedSecondary,
            rhs: &Self::AssignedSecondary,
        ) -> Result<(), Error> {
            self.constrain_equal(layouter, lhs.x(), rhs.x())?;
            self.constrain_equal(layouter, lhs.y(), rhs.y())?;
            self.constrain_equal(layouter, lhs.is_identity(), rhs.is_identity())?;
            Ok(())
        }

        fn assign_constant_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            constant: C::Secondary,
        ) -> Result<Self::AssignedSecondary, Error> {
            let [x, y, is_identity] =
                x_y_is_identity(&constant).map(|value| self.assign_constant(layouter, value));
            Ok(AssignedEcPoint {
                ec_point: Value::known(constant),
                x: x?,
                y: y?,
                is_identity: is_identity?,
            })
        }

        fn assign_witness_secondary(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            witness: Value<C::Secondary>,
        ) -> Result<Self::AssignedSecondary, Error> {
            let collector = &mut self.collector.borrow_mut();
            let zero = collector.register_constant(C::Scalar::ZERO);
            let one = collector.register_constant(C::Scalar::ONE);
            let [x, y, is_identity] = witness
                .as_ref()
                .map(x_y_is_identity)
                .transpose_array()
                .map(|value| collector.new_witness(value));
            collector.assert_bit(&is_identity);
            let not_identity = collector.sub(&one, &is_identity);
            let lhs = collector.mul(&y, &y);
            let lhs = collector.mul(&lhs, &not_identity);
            let x_square_plus_a =
                collector.mul_add_constant_scaled(C::Scalar::ONE, &x, &x, C::Secondary::a());
            let rhs = collector.mul_add_constant_scaled(
                C::Scalar::ONE,
                &x_square_plus_a,
                &x,
                C::Secondary::b(),
            );
            let rhs = collector.mul(&rhs, &not_identity);
            collector.equal(&lhs, &rhs);
            let x = collector.select(&is_identity, &zero, &x);
            let y = collector.select(&is_identity, &zero, &y);
            Ok(AssignedEcPoint {
                ec_point: witness,
                x,
                y,
                is_identity,
            })
        }

        fn assert_if_known_secondary(
            &self,
            value: &Self::AssignedSecondary,
            f: impl FnOnce(&C::Secondary) -> bool,
        ) {
            value.ec_point.assert_if_known(f)
        }

        fn select_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            condition: &Self::Assigned,
            when_true: &Self::AssignedSecondary,
            when_false: &Self::AssignedSecondary,
        ) -> Result<Self::AssignedSecondary, Error> {
            let [x, y, is_identity]: [_; 3] = when_true
                .assigned_cells()
                .zip(when_false.assigned_cells())
                .map(|(when_true, when_false)| {
                    self.select(layouter, condition, when_true, when_false)
                })
                .try_collect::<_, Vec<_>, _>()?
                .try_into()
                .unwrap();
            let output = condition
                .value()
                .zip(when_true.ec_point.zip(when_false.ec_point))
                .map(|(condition, (when_true, when_false))| {
                    if condition == C::Scalar::ONE {
                        when_true
                    } else {
                        when_false
                    }
                });
            Ok(AssignedEcPoint {
                ec_point: output,
                x,
                y,
                is_identity,
            })
        }

        fn add_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            lhs: &Self::AssignedSecondary,
            rhs: &Self::AssignedSecondary,
        ) -> Result<Self::AssignedSecondary, Error> {
            let (out_added, is_x_equal, is_y_equal) = self.add_ec_point_inner(lhs, rhs);
            let out_doubled = self.double_ec_point(lhs);
            let identity = self.assign_constant_secondary(layouter, C::Secondary::identity())?;
            let out = self.select_secondary(layouter, &is_y_equal, &out_doubled, &identity)?;
            self.select_secondary(layouter, &is_x_equal, &out, &out_added)
        }

        fn scalar_mul_secondary(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            base: &Self::AssignedSecondary,
            le_bits: &[Self::Assigned],
        ) -> Result<Self::AssignedSecondary, Error> {
            // TODO
            let mut out = C::Secondary::identity().to_curve();
            for bit in le_bits.iter().rev() {
                bit.value().zip(base.ec_point).map(|(bit, ec_point)| {
                    out = out.double();
                    if bit == C::Scalar::ONE {
                        out += ec_point;
                    }
                });
            }
            self.assign_witness_secondary(layouter, Value::known(out.into()))
        }

        fn fixed_base_msm_secondary<'a, 'b>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            bases: impl IntoIterator<Item = &'a C::Secondary>,
            scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
        ) -> Result<Self::AssignedSecondary, Error>
        where
            Self::AssignedBase: 'b,
        {
            // TODO
            let output = izip_eq!(bases, scalars).fold(
                Value::known(C::Secondary::identity()),
                |acc, (base, scalar)| {
                    acc.zip(scalar.scalar.value())
                        .map(|(acc, scalar)| (acc.to_curve() + *base * scalar).into())
                },
            );
            self.assign_witness_secondary(layouter, output)
        }

        fn variable_base_msm_secondary<'a, 'b>(
            &self,
            layouter: &mut impl Layouter<C::Scalar>,
            bases: impl IntoIterator<Item = &'a Self::AssignedSecondary>,
            scalars: impl IntoIterator<Item = &'b Self::AssignedBase>,
        ) -> Result<Self::AssignedSecondary, Error>
        where
            Self::AssignedSecondary: 'a,
            Self::AssignedBase: 'b,
        {
            // TODO
            let output = izip_eq!(bases, scalars).fold(
                Value::known(C::Secondary::identity()),
                |acc, (base, scalar)| {
                    acc.zip(base.ec_point.zip(scalar.scalar.value()))
                        .map(|(acc, (base, scalar))| (acc.to_curve() + base * scalar).into())
                },
            );
            self.assign_witness_secondary(layouter, output)
        }
    }

    impl<C: TwoChainCurve> HashInstruction<C> for Chip<C>
    where
        C::Base: PrimeFieldBits,
        C::Scalar: FromUniformBytes<64> + PrimeFieldBits,
    {
        const NUM_HASH_BITS: usize = NUM_HASH_BITS;

        type Param = Spec<C::Scalar, T, RATE>;
        type Config = Spec<C::Scalar, T, RATE>;
        type TccChip = Chip<C>;

        fn new(_: Self::Config, chip: Self::TccChip) -> Self {
            chip
        }

        fn hash_state<Comm: AsRef<C::Secondary>>(
            spec: &Self::Param,
            vp_digest: C::Scalar,
            step_idx: usize,
            initial_input: &[C::Scalar],
            output: &[C::Scalar],
            acc: &ProtostarAccumulatorInstance<C::Base, Comm>,
        ) -> C::Scalar {
            let mut poseidon = Poseidon::new_with_spec(spec.clone());
            let fe_to_limbs = |fe| fe_to_limbs(fe, NUM_LIMB_BITS);
            let inputs = iter::empty()
                .chain([vp_digest, C::Scalar::from(step_idx as u64)])
                .chain(initial_input.iter().copied())
                .chain(output.iter().copied())
                .chain(fe_to_limbs(acc.instances[0][0]))
                .chain(fe_to_limbs(acc.instances[0][1]))
                .chain(
                    acc.witness_comms
                        .iter()
                        .map(AsRef::as_ref)
                        .flat_map(x_y_is_identity),
                )
                .chain(acc.challenges.iter().copied().flat_map(fe_to_limbs))
                .chain(fe_to_limbs(acc.u))
                .chain(x_y_is_identity(acc.e_comm.as_ref()))
                .chain(acc.compressed_e_sum.map(fe_to_limbs).into_iter().flatten())
                .collect_vec();
            poseidon.update(&inputs);
            fe_truncated(poseidon.squeeze(), NUM_HASH_BITS)
        }

        fn hash_assigned_state(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            vp_digest: &<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned,
            step_idx: &<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned,
            initial_input: &[<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned],
            output: &[<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned],
            acc: &AssignedProtostarAccumulatorInstance<
                <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedBase,
                <Self::TccChip as TwoChainCurveInstruction<C>>::AssignedSecondary,
            >,
        ) -> Result<<Self::TccChip as TwoChainCurveInstruction<C>>::Assigned, Error> {
            let collector = &mut self.collector.borrow_mut();
            let inputs = iter::empty()
                .chain([vp_digest, step_idx])
                .chain(initial_input)
                .chain(output)
                .chain(acc.instances[0][0].assigned_cells())
                .chain(acc.instances[0][1].assigned_cells())
                .chain(
                    acc.witness_comms
                        .iter()
                        .flat_map(AssignedEcPoint::assigned_cells),
                )
                .chain(acc.challenges.iter().flat_map(AssignedBase::assigned_cells))
                .chain(acc.u.assigned_cells())
                .chain(acc.e_comm.assigned_cells())
                .chain(
                    acc.compressed_e_sum
                        .as_ref()
                        .map(AssignedBase::assigned_cells)
                        .into_iter()
                        .flatten(),
                )
                .copied()
                .collect_vec();
            let mut poseidon_chip = PoseidonChip::from_spec(collector, self.poseidon_spec.clone());
            poseidon_chip.update(&inputs);
            let hash = poseidon_chip.squeeze(collector);
            let hash_le_bits = to_le_bits_strict(collector, &hash);
            Ok(from_le_bits(collector, &hash_le_bits[..NUM_HASH_BITS]))
        }
    }

    #[derive(Clone, Debug)]
    pub struct PoseidonTranscriptChip<C: CurveAffine> {
        poseidon_chip: PoseidonChip<C::Scalar, T, RATE>,
        chip: Chip<C>,
        proof: Value<Cursor<Vec<u8>>>,
    }

    #[derive(Clone)]
    pub struct Challenge<F: PrimeField, N: PrimeField> {
        le_bits: Vec<Witness<N>>,
        scalar: AssignedBase<F, N>,
    }

    impl<F: PrimeField, N: PrimeField> Debug for Challenge<F, N> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut f = f.debug_struct("Challenge");
            self.scalar
                .scalar
                .value()
                .map(|scalar| f.field("scalar", &scalar));
            f.finish()
        }
    }

    impl<F: PrimeField, N: PrimeField> AsRef<AssignedBase<F, N>> for Challenge<F, N> {
        fn as_ref(&self) -> &AssignedBase<F, N> {
            &self.scalar
        }
    }

    impl<C> TranscriptInstruction<C> for PoseidonTranscriptChip<C>
    where
        C: TwoChainCurve,
        C::Base: PrimeFieldBits,
        C::Scalar: FromUniformBytes<64> + PrimeFieldBits,
    {
        type Config = Spec<C::Scalar, T, RATE>;
        type TccChip = Chip<C>;
        type Challenge = Challenge<C::Base, C::Scalar>;

        fn new(spec: Self::Config, chip: Self::TccChip, proof: Value<Vec<u8>>) -> Self {
            let poseidon_chip = PoseidonChip::from_spec(&mut chip.collector.borrow_mut(), spec);
            PoseidonTranscriptChip {
                poseidon_chip,
                chip,
                proof: proof.map(Cursor::new),
            }
        }

        fn challenge_to_le_bits(
            &self,
            _: &mut impl Layouter<C::Scalar>,
            challenge: &Self::Challenge,
        ) -> Result<Vec<Witness<C::Scalar>>, Error> {
            Ok(challenge.le_bits.clone())
        }

        fn common_field_element(
            &mut self,
            value: &AssignedBase<C::Base, C::Scalar>,
        ) -> Result<(), Error> {
            value
                .assigned_cells()
                .for_each(|value| self.poseidon_chip.update(&[*value]));
            Ok(())
        }

        fn common_commitment(
            &mut self,
            value: &AssignedEcPoint<C::Secondary>,
        ) -> Result<(), Error> {
            value
                .assigned_cells()
                .for_each(|value| self.poseidon_chip.update(&[*value]));
            Ok(())
        }

        fn read_field_element(
            &mut self,
            layouter: &mut impl Layouter<C::Scalar>,
        ) -> Result<AssignedBase<C::Base, C::Scalar>, Error> {
            let fe = self.proof.as_mut().and_then(|proof| {
                let mut repr = <C::Base as PrimeField>::Repr::default();
                if proof.read_exact(repr.as_mut()).is_err() {
                    return Value::unknown();
                }
                C::Base::from_repr_vartime(repr)
                    .map(Value::known)
                    .unwrap_or_else(Value::unknown)
            });
            let fe = self.chip.assign_witness_base(layouter, fe)?;
            self.common_field_element(&fe)?;
            Ok(fe)
        }

        fn read_commitment(
            &mut self,
            layouter: &mut impl Layouter<C::Scalar>,
        ) -> Result<AssignedEcPoint<C::Secondary>, Error> {
            let comm = self.proof.as_mut().and_then(|proof| {
                let mut reprs = [<C::Scalar as PrimeField>::Repr::default(); 2];
                for repr in &mut reprs {
                    if proof.read_exact(repr.as_mut()).is_err() {
                        return Value::unknown();
                    }
                }
                let [x, y] = reprs.map(|repr| {
                    C::Scalar::from_repr_vartime(repr)
                        .map(Value::known)
                        .unwrap_or_else(Value::unknown)
                });
                x.zip(y).and_then(|(x, y)| {
                    Option::from(C::Secondary::from_xy(x, y))
                        .map(Value::known)
                        .unwrap_or_else(Value::unknown)
                })
            });
            let comm = self.chip.assign_witness_secondary(layouter, comm)?;
            self.common_commitment(&comm)?;
            Ok(comm)
        }

        fn squeeze_challenge(
            &mut self,
            _: &mut impl Layouter<C::Scalar>,
        ) -> Result<Challenge<C::Base, C::Scalar>, Error> {
            let collector = &mut self.chip.collector.borrow_mut();
            let (challenge_le_bits, challenge) = {
                let hash = self.poseidon_chip.squeeze(collector);
                self.poseidon_chip.update(&[hash]);

                let challenge_le_bits = to_le_bits_strict(collector, &hash)
                    .into_iter()
                    .take(NUM_CHALLENGE_BITS)
                    .collect_vec();
                let challenge = from_le_bits(collector, &challenge_le_bits);

                (challenge_le_bits, challenge)
            };

            let mut integer_chip = IntegerChip::new(collector, &self.chip.rns);
            let limbs = self.chip.rns.from_fe(challenge.value().map(fe_to_fe));
            let scalar = integer_chip.range(limbs, Range::Remainder);
            let limbs = scalar.limbs().iter().map(AsRef::as_ref).copied().collect();

            let scalar_in_base =
                integer_to_native(&self.chip.rns, collector, &scalar, NUM_CHALLENGE_BITS);
            collector.equal(&challenge, &scalar_in_base);

            Ok(Challenge {
                le_bits: challenge_le_bits,
                scalar: AssignedBase { scalar, limbs },
            })
        }
    }
}



