use std::{
    borrow::{Borrow, BorrowMut, Cow},
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    env::set_var,
    fmt::Debug,
    fs::File,
    hash::Hash,
    io::Cursor,
    iter,
    marker::PhantomData,
    rc::Rc,
};

use rand::RngCore;

use halo2_base::{
    gates::flex_gate::{GateChip, GateInstructions},
    utils::{CurveAffineExt, ScalarField},
    halo2_proofs,
    AssignedValue,
    Context,
    QuantumCell::{Constant, Existing, Witness, WitnessFraction},
};

use halo2_proofs::{
    plonk::{
        Advice, Assigned, Circuit, Column, ConstraintSystem, create_proof, Error,
        Fixed, Instance, keygen_pk, keygen_vk, ProvingKey, VerifyingKey, verify_proof,
    },
    circuit::Value,
    halo2curves::{
        bn256::{Bn256, Fr, Fq, G1Affine},
        //group::ff::Field,
        FieldExt,
    },
};

use halo2_ecc::{
    fields::{fp::FpChip, FieldChip, PrimeField},
    bigint::{CRTInteger, FixedCRTInteger, ProperCrtUint},
};

use crate::{
    system::halo2::{compile, transcript::{evm::EvmTranscript, halo2::PoseidonTranscript}, Config},
    loader::{evm::{encode_calldata, Address, EvmLoader, ExecutorBuilder}, halo2, native::NativeLoader, Loader},
    pcs::{Evaluation, kzg::{Gwc19, KzgAs}},
    verifier::{plonk::protocol::{CommonPolynomial, Expression, Query}, SnarkVerifier},
    poly::multilinear::{
        MultilinearPolynomial, rotation_eval_coeff_pattern, rotation_eval_point_pattern, zip_self,
    },
    util::{
        arithmetic::{BooleanHypercube, fe_to_fe, powers, Rotation},
        chain, hash, izip, izip_eq,
        transcript::{Transcript, TranscriptRead, TranscriptWrite},
        BitIndex, DeserializeOwned, Itertools, Serialize,
    },
};


const LIMBS: usize = 3;
const BITS: usize = 88;
const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;
const SECURE_MDS: usize = 0;

// type Poseidon<L> = hash::Poseidon<Fr, L, T, RATE>;
// type BaseFieldEccChip<'chip> = halo2_ecc::ecc::BaseFieldEccChip<'chip, G1Affine>;
// type Halo2Loader<'chip> = halo2::Halo2Loader<G1Affine, BaseFieldEccChip<'chip>>;
//type PoseidonTranscript<L, S> = PoseidonTranscript<G1Affine, L, S, T, RATE, R_F, R_P>;

// check overflow for add/sub_no_carry specially for sum. have done mul with carry everywhere
#[derive(Clone, Debug)]
pub struct Chip<'range, F, CF, GA, L>
where
    CF: PrimeField,
    F: PrimeField,
    GA: CurveAffineExt<Base = CF, ScalarExt = F>,
    L: Loader<GA>,
{
    pub base_chip: &'range FpChip<'range, F, CF>,  
    _marker: PhantomData<(GA, L)>,
}

impl <'range, F, CF, GA, L> Chip<'range, F, CF, GA, L>
    where
    CF: PrimeField,
    F: PrimeField,
    GA: CurveAffineExt<Base = CF, ScalarExt = F>,
    L: Loader<GA>,
{   
    // convert crt to properuint
    // https://github.com/axiom-crypto/halo2-lib/blob/f2eacb1f7fdbb760213cf8037a1bd1a10672133f/halo2-ecc/src/fields/fp.rs#L127

    pub fn new(base_chip: &'range FpChip<'range, F, CF>) -> Self {
        Self {
            base_chip,
            _marker: PhantomData,
        }
    }

    fn load_witnesses<'a>(
        &self,
        ctx: &mut Context<F>,
        witnesses: impl IntoIterator<Item = &'a CF>,
    ) -> Vec<ProperCrtUint<F>> {
            witnesses.into_iter().map(|witness| {self.base_chip.load_private( ctx, witness.clone())}).collect_vec()
    }

    fn powers(
        &self,
        ctx: &mut Context<F>,
        x: &ProperCrtUint<F>,
        n: usize,
    ) -> Result<Vec<ProperCrtUint<F>>, Error> {
        Ok(match n {
            0 => Vec::new(),
            1 => vec![self.base_chip.load_constant(ctx, GA::Base::one())],
            2 => vec![
                self.base_chip.load_constant(ctx, GA::Base::one()),
                x.clone(),
            ],
            _ => {
                let mut powers = Vec::with_capacity(n);
                powers.push(self.base_chip.load_constant(ctx, GA::Base::one()));
                powers.push(x.clone());
                for _ in 0..n - 2 {
                    powers.push(self.base_chip.mul(ctx,powers.last().unwrap(), x));
                }
                powers
            }
        })
    }


    fn inner_product<'a, 'b>(
        &self,
        ctx: &mut Context<F>,
        a: impl IntoIterator<Item = &'a ProperCrtUint<F>>,
        b: impl IntoIterator<Item = &'b ProperCrtUint<F>>,
    ) -> Result<ProperCrtUint<F>, Error> {
        
        let a = a.into_iter();
        let b = b.into_iter();        
        let values = a.zip(b).map(|(a, b)| {
            self.base_chip.mul(ctx, a, b)
        }).collect::<Vec<_>>();

        self.sum(ctx, &values)
    }

    fn sum<'a>(
        &self,
        ctx: &mut Context<F>,
        values: impl IntoIterator<Item = &'a ProperCrtUint<F>>,
    ) -> Result<ProperCrtUint<F>, Error>
    where
        ProperCrtUint<F>: 'a,
    {
        Ok(values.into_iter().fold(
            self.base_chip.load_constant(ctx, GA::Base::zero()),
            |acc, value| 
                FixedCRTInteger::from_native(self.base_chip.add_no_carry(ctx, &acc, value).value.to_biguint().unwrap(), 
                self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                ctx,
                self.base_chip.limb_bits,
                self.base_chip.native_modulus(),
            ),
        ))
    }

    fn product<'a>(
        &self,
        ctx: &mut Context<F>,
        values: impl IntoIterator<Item = &'a ProperCrtUint<F>>,
    ) -> Result<ProperCrtUint<F>, Error>
    where
        ProperCrtUint<F>: 'a,
    {
        Ok(values.into_iter().fold(
            self.base_chip.load_constant(ctx, GA::Base::zero()),
            |acc, value| self.base_chip.mul(ctx, &acc, value),
        ))
    }

    fn horner(
        &self,
        ctx: &mut Context<F>,
        coeffs: &[ProperCrtUint<F>],
        x: &ProperCrtUint<F>,
    ) -> Result<ProperCrtUint<F>, Error> {
        let powers_of_x = self.powers(ctx, x, coeffs.len())?;
        self.inner_product(ctx, coeffs, &powers_of_x)
    }

    // todo too many clones check, if dont use &x get weird long recursive error 
    fn lagrange_and_eval(
        &self,
        ctx: &mut Context<F>,
        coords: &[(ProperCrtUint<F>, ProperCrtUint<F>)],
        x: &ProperCrtUint<F>,
    ) -> Result<ProperCrtUint<F>, Error> {
        assert!(!coords.is_empty(), "coords should not be empty");
        let mut z = self.base_chip.sub_no_carry(ctx, x.clone(), coords[0].0.clone());
        for coord in coords.iter().skip(1) {
            let sub = self.base_chip.sub_no_carry(ctx, x.clone(), coord.0.clone());
            z = self.base_chip.mul(ctx, z, sub).into();
        }
        let mut eval = None;
        for i in 0..coords.len() {
            // compute (x - x_i) * Prod_{j != i} (x_i - x_j)
            // todo which denom not used here -- warning: unused variable: `denom`
            let mut denom = self.base_chip.sub_no_carry(ctx, x.clone(), coords[i].0.clone());
            for j in 0..coords.len() {
                if i == j {
                    continue;
                }
                let sub = self.base_chip.sub_no_carry(ctx, coords[i].0.clone(), coords[j].0.clone());
                let denom = self.base_chip.mul(ctx, denom.clone(), sub.clone());
            }
            let denom = FixedCRTInteger::from_native(denom.clone().value.to_biguint().unwrap(), 
            self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
            ctx,
            self.base_chip.limb_bits,
            self.base_chip.native_modulus());

            let is_zero = self.base_chip.is_zero(ctx, denom.clone());
            // todo check this - primefield doesn't have zero
            self.base_chip.gate().assert_is_const(ctx, &is_zero, &F::zero());

            // y_i / denom
            let quot = self.base_chip.divide_unsafe(ctx, coords[i].1.clone(), denom.clone());
            eval = if let Some(eval) = eval {
                let eval = self.base_chip.add_no_carry(ctx, eval, quot);
                Some(FixedCRTInteger::from_native(eval.value.to_biguint().unwrap(), 
                    self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                    ctx,
                    self.base_chip.limb_bits,
                    self.base_chip.native_modulus()))
            } else {
                Some(quot)
            };
        }
        let out = self.base_chip.mul(ctx, eval.unwrap(), z.clone());
        Ok(out)
    }


    fn rotation_eval_points( 
        &self,
        ctx: &mut Context<F>,
        x: &[ProperCrtUint<F>],
        one_minus_x: &[ProperCrtUint<F>],
        rotation: Rotation,
    ) -> Result<Vec<Vec<ProperCrtUint<F>>>, Error> {
        if rotation == Rotation::cur() {
            return Ok(vec![x.to_vec()]);
        }

        let zero = self.base_chip.load_constant(ctx,GA::Base::zero());
        let one = self.base_chip.load_constant(ctx,GA::Base::one());
        let distance = rotation.distance();
        let num_x = x.len() - distance;
        let points = if rotation < Rotation::cur() {
            let pattern = rotation_eval_point_pattern::<false>(x.len(), distance);
            let x = &x[distance..];
            let one_minus_x = &one_minus_x[distance..];
            pattern
                .iter()
                .map(|pat| {
                    iter::empty()
                        .chain((0..num_x).map(|idx| {
                            if pat.nth_bit(idx) {
                                &one_minus_x[idx]
                            } else {
                                &x[idx]
                            }
                        }))
                        .chain((0..distance).map(|idx| {
                            if pat.nth_bit(idx + num_x) {
                                &one
                            } else {
                                &zero
                            }
                        }))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec()
        } else {
            let pattern = rotation_eval_point_pattern::<true>(x.len(), distance);
            let x = &x[..num_x];
            let one_minus_x = &one_minus_x[..num_x];
            pattern
                .iter()
                .map(|pat| {
                    iter::empty()
                        .chain((0..distance).map(|idx| if pat.nth_bit(idx) { &one } else { &zero }))
                        .chain((0..num_x).map(|idx| {
                            if pat.nth_bit(idx + distance) {
                                &one_minus_x[idx]
                            } else {
                                &x[idx]
                            }
                        }))
                        .cloned()
                        .collect_vec()
                })
                .collect()
            };

        Ok(points)
    }

    fn rotation_eval(
        &self,
        ctx: &mut Context<F>,
        x: &[ProperCrtUint<F>],
        rotation: Rotation,
        evals_for_rotation: &[ProperCrtUint<F>],
    ) -> Result<ProperCrtUint<F>, Error> {
        if rotation == Rotation::cur() {
            assert!(evals_for_rotation.len() == 1);
            return Ok(evals_for_rotation[0].clone());
        }

        let num_vars = x.len();
        let distance = rotation.distance();
        assert!(evals_for_rotation.len() == 1 << distance);
        assert!(distance <= num_vars);

        let (pattern, nths, x) = if rotation < Rotation::cur() {
            (
                rotation_eval_coeff_pattern::<false>(num_vars, distance),
                (1..=distance).rev().collect_vec(),
                x[0..distance].iter().rev().collect_vec(),
            )
        } else {
            (
                rotation_eval_coeff_pattern::<true>(num_vars, distance),
                (num_vars - 1..).take(distance).collect(),
                x[num_vars - distance..].iter().collect(),
            )
        };
        x.into_iter()
            .zip(nths)
            .enumerate()
            .fold(
                Ok(Cow::Borrowed(evals_for_rotation)),
                |evals, (idx, (x_i, nth))| {
                    evals.and_then(|evals| {
                        pattern
                            .iter()
                            .step_by(1 << idx)
                            .map(|pat| pat.nth_bit(nth))
                            .zip(zip_self!(evals.iter()))
                            .map(|(bit, (mut eval_0, mut eval_1))| {
                                if bit {
                                    std::mem::swap(&mut eval_0, &mut eval_1);
                                }
                                let diff = self.base_chip.sub_no_carry(ctx, eval_1, eval_0);
                                let diff_x_i = self.base_chip.mul(ctx, &diff, x_i);
                                
                                Ok(FixedCRTInteger::from_native(self.base_chip.add_no_carry(ctx, &diff_x_i, eval_0).value.to_biguint().unwrap(), 
                                self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                                ctx,
                                self.base_chip.limb_bits,
                                self.base_chip.native_modulus()))

                            })
                            .try_collect::<_, Vec<_>, _>()
                            .map(Into::into)
                    })
                },
            )
            .map(|evals| evals[0].clone())
    }

    fn eq_xy_coeffs(
        &self,
        ctx: &mut Context<F>,
        y: &[ProperCrtUint<F>],
    ) -> Result<Vec<ProperCrtUint<F>>, Error> {
        let mut evals = vec![self.base_chip.load_constant(ctx, GA::Base::one())];

        for y_i in y.iter().rev() {
            evals = evals
                .iter()
                .map(|eval| {
                    let hi = self.base_chip.mul(ctx, eval, y_i);
                    let lo = self.base_chip.sub_no_carry(ctx, eval, &hi);
                    let lo = FixedCRTInteger::from_native(lo.value.to_biguint().unwrap(), 
                    self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                    ctx,
                    self.base_chip.limb_bits,
                    self.base_chip.native_modulus());
                    Ok([lo, hi])
                })
                .try_collect::<_, Vec<_>, Error>()?
                .into_iter()
                .flatten()
                .collect();
        }

        Ok(evals)
    }

    fn eq_xy_eval(
        &self,
        ctx: &mut Context<F>,
        x: &[ProperCrtUint<F>],
        y: &[ProperCrtUint<F>],
    ) -> Result<ProperCrtUint<F>, Error> {
        let terms = izip_eq!(x, y)
            .map(|(x, y)| {
                let one = self.base_chip.load_constant(ctx, GA::Base::one());
                let xy = self.base_chip.mul(ctx, x, y);
                let two_xy = self.base_chip.add_no_carry(ctx, &xy, &xy);
                let two_xy_plus_one = self.base_chip.add_no_carry(ctx, &two_xy, &one);
                let x_plus_y = self.base_chip.add_no_carry(ctx, x, y);
                Ok(FixedCRTInteger::from_native(self.base_chip.sub_no_carry(ctx, &two_xy_plus_one, &x_plus_y).value.to_biguint().unwrap(), 
                                self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                                ctx,
                                self.base_chip.limb_bits,
                                self.base_chip.native_modulus()))
            })
            .try_collect::<_, Vec<_>, Error>()?;
        self.product(ctx, &terms)
    }

    // #[allow(clippy::too_many_arguments)]
    // fn evaluate(
    //     &self,
    //     ctx: &mut Context<F>,
    //     expression: &Expression<F>,
    //     identity_eval: &ProperCrtUint<F>,
    //     lagrange_evals: &BTreeMap<i32, ProperCrtUint<F>>,
    //     eq_xy_eval: &ProperCrtUint<F>,
    //     query_evals: &BTreeMap<Query, ProperCrtUint<F>>,
    //     challenges: &[ProperCrtUint<F>],
    // ) -> Result<ProperCrtUint<F>, Error> {
    //     let mut evaluate = |expression| {
    //         self.evaluate(
    //             ctx,
    //             expression,
    //             identity_eval,
    //             lagrange_evals,
    //             eq_xy_eval,
    //             query_evals,
    //             challenges,
    //         )
    //     };
    //     match expression {
    //         Expression::Constant(scalar) => Ok(self.base_chip.load_constant(ctx,*scalar)),
    //         Expression::CommonPolynomial(poly) => match poly {
    //             CommonPolynomial::Identity => Ok(identity_eval.clone()),
    //             CommonPolynomial::Lagrange(i) => Ok(lagrange_evals[i].clone()),
    //             CommonPolynomial::EqXY(idx) => {
    //                 assert_eq!(*idx, 0);
    //                 Ok(eq_xy_eval.clone())
    //             }
    //         },
    //         Expression::Polynomial(query) => Ok(query_evals[query].clone()),
    //         Expression::Challenge(index) => Ok(challenges[*index].clone()),
    //         Expression::Negated(a) => {
    //             let a = evaluate(a)?;
    //             Ok(self.base_chip.neg(ctx, &a))
    //         }
    //         Expression::Sum(a, b) => {
    //             let a = evaluate(a)?;
    //             let b = evaluate(b)?;
    //             Ok(self.base_chip.add_no_carry(ctx, &a, &b))
    //         }
    //         Expression::Product(a, b) => {
    //             let a = evaluate(a)?;
    //             let b = evaluate(b)?;
    //             Ok(self.base_chip.mul(ctx, &a, &b))
    //         }
    //         Expression::Scaled(a, scalar) => {
    //             let a = evaluate(a)?;
    //             let scalar = self.base_chip.load_constant(ctx,*scalar)?;
    //             Ok(self.base_chip.mul(ctx, &a, &scalar))
    //         }
    //         Expression::DistributePowers(exprs, scalar) => {
    //             assert!(!exprs.is_empty());
    //             if exprs.len() == 1 {
    //                 return evaluate(&exprs[0]);
    //             }
    //             let scalar = evaluate(scalar)?;
    //             let exprs = exprs.iter().map(evaluate).try_collect::<_, Vec<_>, _>()?;
    //             let mut scalars = Vec::with_capacity(exprs.len());
    //             scalars.push(self.base_chip.load_constant(ctx,GA::Base::one())?);
    //             scalars.push(scalar);
    //             for _ in 2..exprs.len() {
    //                 scalars.push(self.base_chip.mul(ctx, &scalars[1], scalars.last().unwrap())?);
    //             }
    //             Ok(self.inner_product(ctx, &scalars, &exprs))
    //         }
    //     }
    // }

    fn verify_sum_check<const IS_MSG_EVALS: bool,T>(
        &self,
        ctx: &mut Context<F>,
        num_vars: usize,
        degree: usize,
        sum: &ProperCrtUint<F>,
        transcript: &mut T // impl TranscriptInstruction<F, TccChip = Self>,
    ) -> Result<(ProperCrtUint<F>, Vec<ProperCrtUint<F>>), Error> 
    // fix add loader here
    where T: TranscriptRead<GA,L>
    {
        let points = iter::successors(Some(GA::Base::zero()), move |state| Some(GA::Base::one() + state)).take(degree + 1).collect_vec();
        let points = points
        .into_iter()
        .map(|point| 
            Ok(self.base_chip.load_private(ctx, point)))
            .try_collect::<_, Vec<_>, Error>()?;

        let mut sum = Cow::Borrowed(sum);
        let mut x = Vec::with_capacity(num_vars);
        //let mut transcript = PoseidonTranscript::<NativeLoader, _>; 
        for _ in 0..num_vars{
            // let msg = transcript.read_n_scalars(degree + 1);
            // x.push(transcript.squeeze_challenge().as_ref().clone());

            let msg = Vec::with_capacity(num_vars);
            let sum_from_evals = if IS_MSG_EVALS {
                FixedCRTInteger::from_native(self.base_chip.add_no_carry(ctx, &msg[0], &msg[1]).value.to_biguint().unwrap(), 
                self.base_chip.num_limbs, self.base_chip.limb_bits).assign(
                ctx,
                self.base_chip.limb_bits,
                self.base_chip.native_modulus())
            } else {
                self.sum(ctx, chain![[&msg[0], &msg[0]], &msg[1..]]).unwrap()
            };
            self.base_chip.assert_equal( ctx, &*sum, &sum_from_evals);

            let coords:Vec<_> = points
            .iter()
            .cloned()
            .zip(msg.iter().cloned())
            .collect();

            if IS_MSG_EVALS {
                sum = Cow::Owned(self.lagrange_and_eval(
                    ctx,
                    &coords,
                    &x.last().unwrap(),
                ).unwrap());
            } else {
                sum = Cow::Owned(self.horner(ctx, &msg, &x.last().unwrap()).unwrap());
            };
        }

        Ok((sum.into_owned(), x))
    }

    // #[allow(clippy::too_many_arguments)]
    // #[allow(clippy::type_complexity)]
    // fn verify_sum_check_and_query(
    //     &self,
    //     ctx: &mut Context<F>,
    //     num_vars: usize,
    //     expression: &Expression<F>,
    //     sum: &ProperCrtUint<F>,
    //     instances: &[Vec<ProperCrtUint<F>>],
    //     challenges: &[ProperCrtUint<F>],
    //     y: &[ProperCrtUint<F>],
    //     transcript: &mut impl TranscriptInstruction<F, TccChip = Self>,
    // ) -> Result<
    //     (
    //         Vec<Vec<ProperCrtUint<F>>>,
    //         Vec<Evaluation<ProperCrtUint<F>>>,
    //     ),
    //     Error,
    // > {
    //     let degree = expression.degree();

    //     let (x_eval, x) =
    //         self.verify_sum_check::<true>( ctx, num_vars, degree, sum, transcript)?;

    //     let pcs_query = {
    //         let mut used_query = expression.used_query();
    //         used_query.retain(|query| query.poly() >= instances.len());
    //         used_query
    //     };
    //     let (evals_for_rotation, query_evals) = pcs_query
    //         .iter()
    //         .map(|query| {
    //             let evals_for_rotation =
    //                 transcript.read_field_elements( 1 << query.rotation().distance())?;
    //             let eval = self.rotation_eval(
    //                 ctx,
    //                 x.as_ref(),
    //                 query.rotation(),
    //                 &evals_for_rotation,
    //             )?;
    //             Ok((evals_for_rotation, (*query, eval)))
    //         })
    //         .try_collect::<_, Vec<_>, Error>()?
    //         .into_iter()
    //         .unzip::<_, _, Vec<_>, Vec<_>>();

    //     let one = self.base_chip.load_constant(ctx,GA::Base::one())?;
    //     let one_minus_x = x
    //         .iter()
    //         .map(|x_i| self.base_chip.sub_no_carry( ctx, &one, x_i))
    //         .try_collect::<_, Vec<_>, _>()?;

    //     let (lagrange_evals, query_evals) = {
    //         let mut instance_query = expression.used_query();
    //         instance_query.retain(|query| query.poly() < instances.len());

    //         let lagranges = {
    //             let mut lagranges = instance_query.iter().fold(0..0, |range, query| {
    //                 let i = -query.rotation().0;
    //                 range.start.min(i)..range.end.max(i + instances[query.poly()].len() as i32)
    //             });
    //             if lagranges.start < 0 {
    //                 lagranges.start -= 1;
    //             }
    //             if lagranges.end > 0 {
    //                 lagranges.end += 1;
    //             }
    //             chain![lagranges, expression.used_langrange()].collect::<BTreeSet<_>>()
    //         };

    //         let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
    //         let lagrange_evals = lagranges
    //             .into_iter()
    //             .map(|i| {
    //                 let b = bh[i.rem_euclid(1 << num_vars as i32) as usize];
    //                 let eval = self.product(
                        
    //                     (0..num_vars).map(|idx| {
    //                         if b.nth_bit(idx) {
    //                             &x[idx]
    //                         } else {
    //                             &one_minus_x[idx]
    //                         }
    //                     }),
    //                 )?;
    //                 Ok((i, eval))
    //             })
    //             .try_collect::<_, BTreeMap<_, _>, Error>()?;

    //         let instance_evals = instance_query
    //             .into_iter()
    //             .map(|query| {
    //                 let is = if query.rotation() > Rotation::cur() {
    //                     (-query.rotation().0..0)
    //                         .chain(1..)
    //                         .take(instances[query.poly()].len())
    //                         .collect_vec()
    //                 } else {
    //                     (1 - query.rotation().0..)
    //                         .take(instances[query.poly()].len())
    //                         .collect_vec()
    //                 };
    //                 let eval = self.inner_product(
    //                     ctx,
    //                     &instances[query.poly()],
    //                     is.iter().map(|i| lagrange_evals.get(i).unwrap()),
    //                 )?;
    //                 Ok((query, eval))
    //             })
    //             .try_collect::<_, BTreeMap<_, _>, Error>()?;

    //         (
    //             lagrange_evals,
    //             chain![query_evals, instance_evals].collect(),
    //         )
    //     };
    //     let identity_eval = {
    //         let powers_of_two = powers(GA::Base::one().double())
    //             .take(x.len())
    //             .map(|power_of_two| self.base_chip.load_constant(ctx,power_of_two))
    //             .try_collect::<_, Vec<_>, Error>()?;
    //         self.inner_product(ctx, &powers_of_two, &x)?
    //     };
    //     let eq_xy_eval = self.eq_xy_eval(ctx, &x, y)?;

    //     let eval = self.evaluate(
    //         ctx,
    //         expression,
    //         &identity_eval,
    //         &lagrange_evals,
    //         &eq_xy_eval,
    //         &query_evals,
    //         challenges,
    //     )?;
    //     ctx.constrain_equal(&x_eval, &eval)?;

    //     let points = pcs_query
    //         .iter()
    //         .map(Query::rotation)
    //         .collect::<BTreeSet<_>>()
    //         .into_iter()
    //         .map(|rotation| self.rotation_eval_points(ctx, &x, &one_minus_x, rotation))
    //         .try_collect::<_, Vec<_>, _>()?
    //         .into_iter()
    //         .flatten()
    //         .collect_vec();
    //     // add this point offset fn from hyperplonk backend or implement in halo2 like points and pcs query
    //     let point_offset = point_offset(&pcs_query);
    //     let evals = pcs_query
    //         .iter()
    //         .zip(evals_for_rotation)
    //         .flat_map(|(query, evals_for_rotation)| {
    //             (point_offset[&query.rotation()]..)
    //                 .zip(evals_for_rotation)
    //                 .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
    //         })
    //         .collect();
    //     Ok((points, evals))
    // }

    // look into this
    // #[allow(clippy::type_complexity)]
    // fn multilinear_pcs_batch_verify<'a, Comm>(
    //     &self,
    //     ctx: &mut Context<F>,
    //     comms: &'a [Comm],
    //     points: &[Vec<ProperCrtUint<F>>],
    //     evals: &[Evaluation<ProperCrtUint<F>>],
    //     transcript: &mut impl TranscriptInstruction<F, TccChip = Self>,
    // ) -> Result<
    //     (
    //         Vec<(&'a Comm, ProperCrtUint<F>)>,
    //         Vec<ProperCrtUint<F>>,
    //         ProperCrtUint<F>,
    //     ),
    //     Error,
    // > {
    //     let num_vars = points[0].len();

    //     let ell = evals.len().next_power_of_two().ilog2() as usize;
    //     let t = transcript
    //         .squeeze_challenges( ell)?
    //         .iter()
    //         .map(AsRef::as_ref)
    //         .cloned()
    //         .collect_vec();

    //     let eq_xt = self.eq_xy_coeffs(ctx, &t)?;
    //     let tilde_gs_sum = self.inner_product(
    //         ctx,
    //         &eq_xt[..evals.len()],
    //         evals.iter().map(Evaluation::value),
    //     )?;
    //     let (g_prime_eval, x) =
    //         self.verify_sum_check::<false>(ctx, num_vars, 2, &tilde_gs_sum, transcript)?;
    //     let eq_xy_evals = points
    //         .iter()
    //         .map(|point| self.eq_xy_eval(ctx, &x, point))
    //         .try_collect::<_, Vec<_>, _>()?;

    //     let g_prime_comm = {
    //         let scalars = evals.iter().zip(&eq_xt).fold(
    //             Ok::<_, Error>(BTreeMap::<_, _>::new()),
    //             |scalars, (eval, eq_xt_i)| {
    //                 let mut scalars = scalars?;
    //                 let scalar = self.base_chip.mul(ctx, &eq_xy_evals[eval.point()], eq_xt_i)?;
    //                 match scalars.entry(eval.poly()) {
    //                     Entry::Occupied(mut entry) => {
    //                         *entry.get_mut() = self.base_chip.add_no_carry(ctx, entry.get(), &scalar)?;
    //                     }
    //                     Entry::Vacant(entry) => {
    //                         entry.insert(scalar);
    //                     }
    //                 }
    //                 Ok(scalars)
    //             },
    //         )?;
    //         scalars
    //             .into_iter()
    //             .map(|(poly, scalar)| (&comms[poly], scalar))
    //             .collect_vec()
    //     };

    //     Ok((g_prime_comm, x, g_prime_eval))
    // }

    // todo change these 3 fns to verify_hyperplonk_gemini_kzg - used by protostar prover
    // todo change self.add(a,b) and other similar fns with self.base_chip.add_no_carry(ctx,a,b)
    // fn verify_ipa<'a>(
    //     &self,
    //     ctx: &mut Context<F>,
    //     vp: &MultilinearIpaParams<C::Secondary>,
    //     comm: impl IntoIterator<Item = (&'a Self::AssignedSecondary, &'a ProperCrtUint<F>)>,
    //     point: &[ProperCrtUint<F>],
    //     eval: &ProperCrtUint<F>,
    //     transcript: &mut impl TranscriptInstruction<F, TccChip = Self>,
    // ) -> Result<(), Error>
    // where
    //     Self::AssignedSecondary: 'a,
    //     ProperCrtUint<F>: 'a,
    // {
    //     let xi_0 = transcript.squeeze_challenge()?.as_ref().clone();

    //     let (ls, rs, xis) = iter::repeat_with(|| {
    //         Ok::<_, Error>((
    //             transcript.read_commitment()?,
    //             transcript.read_commitment()?,
    //             transcript.squeeze_challenge()?.as_ref().clone(),
    //         ))
    //     })
    //     .take(point.len())
    //     .try_collect::<_, Vec<_>, _>()?
    //     .into_iter()
    //     .multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
    //     let g_k = transcript.read_commitment()?;
    //     let c = transcript.read_field_element()?;

    //     let xi_invs = xis
    //         .iter()
    //         .map(|xi| self.invert_incomplete( xi))
    //         .try_collect::<_, Vec<_>, _>()?;
    //     let eval_prime = self.mul( &xi_0, eval)?;

    //     let h_eval = {
    //         let one = self.base_chip.load_constant(ctx, GA::Base::one())?;
    //         let terms = izip_eq!(point, xis.iter().rev())
    //             .map(|(point, xi)| {
    //                 let point_xi = self.mul( point, xi)?;
    //                 let neg_point = self.neg( point)?;
    //                 self.sum( ctx, [&one, &neg_point, &point_xi])
    //             })
    //             .try_collect::<_, Vec<_>, _>()?;
    //         self.product( &terms)?
    //     };
    //     let h_coeffs = {
    //         let one = self.base_chip.load_constant(ctx, GA::Base::one())?;
    //         let mut coeff = vec![one];

    //         for xi in xis.iter().rev() {
    //             let extended = coeff
    //                 .iter()
    //                 .map(|coeff| self.mul( coeff, xi))
    //                 .try_collect::<_, Vec<_>, _>()?;
    //             coeff.extend(extended);
    //         }

    //         coeff
    //     };

    //     let neg_c = self.neg( &c)?;
    //     let h_scalar = {
    //         let mut tmp = self.mul( &neg_c, &h_eval)?;
    //         tmp = self.mul( &tmp, &xi_0)?;
    //         self.add( &tmp, &eval_prime)?
    //     };
    //     let range = RangeChip::<C>::default(lookup_bits);
    //     let fp_chip = FpChip::<C>::new(&range, BITS, LIMBS);
    //     let ecc_chip = BaseFieldEccChip::new(&fp_chip);
    //     // todo find similar to C::Secondary::identity() in Fr
    //     let identity = ecc_chip.assign_constant( C::Secondary::identity())?;
    //     let out = {
    //         let h = ecc_chip.assign_constant( *vp.h())?;
    //         let (mut bases, mut scalars) = comm.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
    //         bases.extend(chain![&ls, &rs, [&h, &g_k]]);
    //         scalars.extend(chain![&xi_invs, &xis, [&h_scalar, &neg_c]]);
    //         // todo change the inputs in form of a tuple
    //         ecc_chip.variable_base_msm( ctx,(bases, scalars))?
    //     };
    //     // is this equal to assert_equal in shim.rs? 
    //     ecc_chip.constrain_equal_secondary( &out, &identity)?;

    //     let out = {
    //         let bases = vp.g();
    //         let scalars = h_coeffs;
    //         // todo change the inputs in form of a tuple
    //         ecc_chip.fixed_base_msm( bases, &scalars)?
    //     };
    //     ecc_chip.constrain_equal_secondary( &out, &g_k)?;

    //     Ok(())
    // }


    // fn verify_hyrax(
    //     &self,
    //     ctx: &mut Context<F>,
    //     vp: &MultilinearHyraxParams<C::Secondary>,
    //     comm: &[(&Vec<Self::AssignedSecondary>, ProperCrtUint<F>)], // &[(&Vec<EcPoint<F, ProperCrtUint<F>>, ProperCrtUint<F>)]
    //     point: &[ProperCrtUint<F>],
    //     eval: &ProperCrtUint<F>,
    //     transcript: &mut impl TranscriptInstruction<F, TccChip = Self>,
    // ) -> Result<(), Error> {
    //     let (lo, hi) = point.split_at(vp.row_num_vars());
    //     let scalars = self.eq_xy_coeffs(ctx, hi)?;

    //     let comm = comm
    //         .iter()
    //         .map(|(comm, rhs)| {
    //             let scalars = scalars
    //                 .iter()
    //                 .map(|lhs| self.mul( lhs, rhs))
    //                 .try_collect::<_, Vec<_>, _>()?;
    //             Ok::<_, Error>(izip_eq!(*comm, scalars))
    //         })
    //         .try_collect::<_, Vec<_>, _>()?
    //         .into_iter()
    //         .flatten()
    //         .collect_vec();
    //     let comm = comm.iter().map(|(comm, scalar)| (*comm, scalar));

    //     self.verify_ipa(ctx, vp.ipa(), comm, lo, eval, transcript)
    // }

    // fn verify_gemini_hyperplonk(
    //     &self,
    //     ctx: &mut Context<F>,
    //     vp: &HyperPlonkVerifierParam<F, MultilinearHyrax<C::Secondary>>,
    //     instances: Value<&[F]>,
    //     transcript: &mut impl TranscriptInstruction<F, TccChip = Self>,
    // ) -> Result<(), Error>
    // where
    //     F: Serialize + DeserializeOwned,
    //     C::Secondary: Serialize + DeserializeOwned,
    // {
    //     assert_eq!(vp.num_instances.len(), 1);
    //     let instances = vec![instances
    //         .transpose_vec(vp.num_instances[0])
    //         .into_iter()
    //         .map(|instance| self.assign_witness( instance.copied()))
    //         .try_collect::<_, Vec<_>, _>()?];

    //     transcript.common_field_elements(&instances[0])?;

    //     let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
    //     let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 3);
    //     for (num_polys, num_challenges) in
    //         vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
    //     {
    //         witness_comms.extend(
    //             iter::repeat_with(|| transcript.read_commitments( vp.pcs.num_chunks()))
    //                 .take(*num_polys)
    //                 .try_collect::<_, Vec<_>, _>()?,
    //         );
    //         challenges.extend(
    //             transcript
    //                 .squeeze_challenges( *num_challenges)?
    //                 .iter()
    //                 .map(AsRef::as_ref)
    //                 .cloned(),
    //         );
    //     }

    //     let beta = transcript.squeeze_challenge()?.as_ref().clone();

    //     let lookup_m_comms =
    //         iter::repeat_with(|| transcript.read_commitments( vp.pcs.num_chunks()))
    //             .take(vp.num_lookups)
    //             .try_collect::<_, Vec<_>, _>()?;

    //     let gamma = transcript.squeeze_challenge()?.as_ref().clone();

    //     let lookup_h_permutation_z_comms =
    //         iter::repeat_with(|| transcript.read_commitments( vp.pcs.num_chunks()))
    //             .take(vp.num_lookups + vp.num_permutation_z_polys)
    //             .try_collect::<_, Vec<_>, _>()?;

    //     let alpha = transcript.squeeze_challenge()?.as_ref().clone();
    //     let y = transcript
    //         .squeeze_challenges( vp.num_vars)?
    //         .iter()
    //         .map(AsRef::as_ref)
    //         .cloned()
    //         .collect_vec();

    //     challenges.extend([beta, gamma, alpha]);

    //     let zero = self.base_chip.load_constant(ctx,GA::Base::zero())?;
    //     let (points, evals) = self.verify_sum_check_and_query(
    //         ctx,
    //         vp.num_vars,
    //         &vp.expression,
    //         &zero,
    //         &instances,
    //         &challenges,
    //         &y,
    //         transcript,
    //     )?;

    //     let range = RangeChip::<Fr>::default(lookup_bits);
    //     let fp_chip = FpChip::<Fr>::new(&range, BITS, LIMBS);
    //     let ecc_chip = BaseFieldEccChip::new(&fp_chip);

    //     let dummy_comm = vec![
    //         ecc_chip.assign_constant( C::Secondary::identity())?;
    //         vp.pcs.num_chunks()
    //     ];
    //     let preprocess_comms = vp
    //         .preprocess_comms
    //         .iter()
    //         .map(|comm| {
    //             comm.0
    //                 .iter()
    //                 .map(|c| ecc_chip.assign_constant( *c))
    //                 .try_collect::<_, Vec<_>, _>()
    //         })
    //         .try_collect::<_, Vec<_>, _>()?;
    //     let permutation_comms = vp
    //         .permutation_comms
    //         .iter()
    //         .map(|comm| {
    //             comm.1
    //                  .0
    //                 .iter()
    //                 .map(|c| ecc_chip.assign_constant( *c))
    //                 .try_collect::<_, Vec<_>, _>()
    //         })
    //         .try_collect::<_, Vec<_>, _>()?;
    //     let comms = iter::empty()
    //         .chain(iter::repeat(dummy_comm).take(vp.num_instances.len()))
    //         .chain(preprocess_comms)
    //         .chain(witness_comms)
    //         .chain(permutation_comms)
    //         .chain(lookup_m_comms)
    //         .chain(lookup_h_permutation_z_comms)
    //         .collect_vec();

    //     let (comm, point, eval) =
    //         self.multilinear_pcs_batch_verify(ctx, &comms, &points, &evals, transcript)?;

    //     self.verify_gemini(ctx, &vp.pcs, &comm, &point, &eval, transcript)?;

    //     Ok(())
    // }

}

#[cfg(test)]
mod test {
// External crates
use halo2_base::halo2_proofs::{
    arithmetic::CurveAffine,
    dev::MockProver,
    halo2curves::bn256::{self, Fr, Fq}, plonk::Assigned,
};
use halo2_base::gates::{
    builder::{
        CircuitBuilderStage, GateThreadBuilder, MultiPhaseThreadBreakPoints, RangeCircuitBuilder,
    },
    RangeChip,
};
use halo2_base::utils::{biguint_to_fe, fe_to_biguint, modulus, CurveAffineExt, ScalarField};
use halo2_base::{Context, AssignedValue};
use halo2_ecc::{
    bn254::{FpChip,FpPoint},
    fields::{FieldChip, PrimeField},
    bigint::{CRTInteger, FixedCRTInteger, ProperCrtUint},
};
// Current crate and module
use crate::loader::{
    evm::{encode_calldata, Address, EvmLoader, ExecutorBuilder},
    halo2,
    native::NativeLoader,
    Loader,
};

use test_case::test_case;
use std::{env::var, fs::File, io::Cursor, rc::Rc};

// Current module
use super::Chip;

const LIMBS: usize = 3;
const BITS: usize = 88;

// RUSTFLAGS="-A warnings" cargo test --package snark-verifier  --lib -- protostar::verifier::test::test_lagrange_eval::lagrange_eval_constant_fn --exact --nocapture

#[test_case(&[0, 1, 2].map(Fq::from) => (Fr::one()) ; "lagrange_eval(): constant fn")]
pub fn test_lagrange_eval (input: &[Fq]) -> (Fr) {
    let mut builder = GateThreadBuilder::mock();
    let ctx = builder.main(0);
    //var("LOOKUP_BITS").unwrap_or_else(|_| panic!("LOOKUP_BITS not set")).parse().unwrap();
    let range = RangeChip::default(8);
    let fp_chip = FpChip::new(&range, BITS, LIMBS); 
    let chip: Chip<_, _, bn256::G1Affine, NativeLoader> = Chip::new(&fp_chip);   
    
    let input = chip.load_witnesses(ctx, input.clone());
    let a = chip.lagrange_and_eval(ctx, &[(input[0].clone(), input[1].clone())], &input[2]);

    println!("{:?}", *a.as_ref().unwrap().native().value());
    *a.unwrap().native().value()
    
}

// #[test_case((vec![Witness(Fr::one()); 5], vec![Witness(Fr::one()); 5]) => Fr::from(5) ; "inner_product(): 1 * 1 + ... + 1 * 1 == 5")]
// pub fn test_inner_product<F: ScalarField>(input: (Vec<QuantumCell<F>>, Vec<QuantumCell<F>>)) -> F {
//     let mut builder = GateThreadBuilder::mock();
//     let ctx = builder.main(0);
//     let chip = GateChip::default();
//     let a = chip.inner_product(ctx, input.0, input.1);
//     *a.value()
// }

}


// #[cfg(test)]
// pub(super) mod test {
//     use crate::{
//         piop::sum_check::{evaluate, SumCheck, VirtualPolynomial},
//         poly::multilinear::{rotation_eval, MultilinearPolynomial},
//         util::{
//             expression::Expression,
//             transcript::{InMemoryTranscript, Keccak256Transcript},
//         },
//     };
//     use halo2_curves::bn256::Fr;
//     use std::ops::Range;

//     pub fn run_sum_check<S: SumCheck<Fr>>(
//         num_vars_range: Range<usize>,
//         expression_fn: impl Fn(usize) -> Expression<Fr>,
//         param_fn: impl Fn(usize) -> (S::ProverParam, S::VerifierParam),
//         assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Fr>),
//         sum: Fr,
//     ) {
//         for num_vars in num_vars_range {
//             let expression = expression_fn(num_vars);
//             let degree = expression.degree();
//             let (pp, vp) = param_fn(expression.degree());
//             let (polys, challenges, y) = assignment_fn(num_vars);
//             let ys = [y];
//             let proof = {
//                 let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
//                 let mut transcript = Keccak256Transcript::default();
//                 S::prove(&pp, num_vars, virtual_poly, sum, &mut transcript).unwrap();
//                 transcript.into_proof()
//             };
//             let accept = {
//                 let mut transcript = Keccak256Transcript::from_proof((), proof.as_slice());
//                 let (x_eval, x) =
//                     S::verify(&vp, num_vars, degree, Fr::zero(), &mut transcript).unwrap();
//                 let evals = expression
//                     .used_query()
//                     .into_iter()
//                     .map(|query| {
//                         let evaluate_for_rotation =
//                             polys[query.poly()].evaluate_for_rotation(&x, query.rotation());
//                         let eval = rotation_eval(&x, query.rotation(), &evaluate_for_rotation);
//                         (query, eval)
//                     })
//                     .collect();
//                 x_eval == evaluate(&expression, num_vars, &evals, &challenges, &[&ys[0]], &x)
//             };
//             assert!(accept);
//         }
//     }

//     pub fn run_zero_check<S: SumCheck<Fr>>(
//         num_vars_range: Range<usize>,
//         expression_fn: impl Fn(usize) -> Expression<Fr>,
//         param_fn: impl Fn(usize) -> (S::ProverParam, S::VerifierParam),
//         assignment_fn: impl Fn(usize) -> (Vec<MultilinearPolynomial<Fr>>, Vec<Fr>, Vec<Fr>),
//     ) {
//         run_sum_check::<S>(
//             num_vars_range,
//             expression_fn,
//             param_fn,
//             assignment_fn,
//             Fr::zero(),
//         )
//     }

//     macro_rules! tests {
//         ($impl:ty) => {
//             #[test]
//             fn sum_check_lagrange() {
//                 use halo2_curves::bn256::Fr;
//                 use $crate::{
//                     piop::sum_check::test::run_zero_check,
//                     poly::multilinear::MultilinearPolynomial,
//                     util::{
//                         arithmetic::{BooleanHypercube, Field},
//                         expression::{CommonPolynomial, Expression, Query, Rotation},
//                         test::{rand_vec, seeded_std_rng},
//                         Itertools,
//                     },
//                 };

//                 run_zero_check::<$impl>(
//                     2..4,
//                     |num_vars| {
//                         let polys = (0..1 << num_vars)
//                             .map(|idx| {
//                                 Expression::<Fr>::Polynomial(Query::new(idx, Rotation::cur()))
//                             })
//                             .collect_vec();
//                         let gates = polys
//                             .iter()
//                             .enumerate()
//                             .map(|(i, poly)| {
//                                 Expression::CommonPolynomial(CommonPolynomial::Lagrange(i as i32))
//                                     - poly
//                             })
//                             .collect_vec();
//                         let alpha = Expression::Challenge(0);
//                         let eq = Expression::eq_xy(0);
//                         Expression::distribute_powers(&gates, &alpha) * eq
//                     },
//                     |_| ((), ()),
//                     |num_vars| {
//                         let polys = BooleanHypercube::new(num_vars)
//                             .iter()
//                             .map(|idx| {
//                                 let mut polys =
//                                     MultilinearPolynomial::new(vec![Fr::zero(); 1 << num_vars]);
//                                 polys[idx] = Fr::one();
//                                 polys
//                             })
//                             .collect_vec();
//                         let alpha = Fr::random(seeded_std_rng());
//                         (polys, vec![alpha], rand_vec(num_vars, seeded_std_rng()))
//                     },
//                 );
//             }

//             #[test]
//             fn sum_check_rotation() {
//                 use halo2_curves::bn256::Fr;
//                 use std::iter;
//                 use $crate::{
//                     piop::sum_check::test::run_zero_check,
//                     poly::multilinear::MultilinearPolynomial,
//                     util::{
//                         arithmetic::{BooleanHypercube, Field},
//                         expression::{Expression, Query, Rotation},
//                         test::{rand_vec, seeded_std_rng},
//                         Itertools,
//                     },
//                 };

//                 run_zero_check::<$impl>(
//                     2..16,
//                     |num_vars| {
//                         let polys = (-(num_vars as i32) + 1..num_vars as i32)
//                             .rev()
//                             .enumerate()
//                             .map(|(idx, rotation)| {
//                                 Expression::<Fr>::Polynomial(Query::new(idx, rotation.into()))
//                             })
//                             .collect_vec();
//                         let gates = polys
//                             .windows(2)
//                             .map(|polys| &polys[1] - &polys[0])
//                             .collect_vec();
//                         let alpha = Expression::Challenge(0);
//                         let eq = Expression::eq_xy(0);
//                         Expression::distribute_powers(&gates, &alpha) * eq
//                     },
//                     |_| ((), ()),
//                     |num_vars| {
//                         let bh = BooleanHypercube::new(num_vars);
//                         let rotate = |f: &Vec<Fr>| {
//                             (0..1 << num_vars)
//                                 .map(|idx| f[bh.rotate(idx, Rotation::next())])
//                                 .collect_vec()
//                         };
//                         let poly = rand_vec(1 << num_vars, seeded_std_rng());
//                         let polys = iter::successors(Some(poly), |poly| Some(rotate(poly)))
//                             .map(MultilinearPolynomial::new)
//                             .take(2 * num_vars - 1)
//                             .collect_vec();
//                         let alpha = Fr::random(seeded_std_rng());
//                         (polys, vec![alpha], rand_vec(num_vars, seeded_std_rng()))
//                     },
//                 );
//             }

//             #[test]
//             fn sum_check_vanilla_plonk() {
//                 use halo2_curves::bn256::Fr;
//                 use $crate::{
//                     backend::hyperplonk::util::{
//                         rand_vanilla_plonk_assignment, vanilla_plonk_expression,
//                     },
//                     piop::sum_check::test::run_zero_check,
//                     util::test::{rand_vec, seeded_std_rng},
//                 };

//                 run_zero_check::<$impl>(
//                     2..16,
//                     |num_vars| vanilla_plonk_expression(num_vars),
//                     |_| ((), ()),
//                     |num_vars| {
//                         let (polys, challenges) = rand_vanilla_plonk_assignment(
//                             num_vars,
//                             seeded_std_rng(),
//                             seeded_std_rng(),
//                         );
//                         (polys, challenges, rand_vec(num_vars, seeded_std_rng()))
//                     },
//                 );
//             }

//             #[test]
//             fn sum_check_vanilla_plonk_with_lookup() {
//                 use halo2_curves::bn256::Fr;
//                 use $crate::{
//                     backend::hyperplonk::util::{
//                         rand_vanilla_plonk_with_lookup_assignment,
//                         vanilla_plonk_with_lookup_expression,
//                     },
//                     piop::sum_check::test::run_zero_check,
//                     util::test::{rand_vec, seeded_std_rng},
//                 };

//                 run_zero_check::<$impl>(
//                     2..16,
//                     |num_vars| vanilla_plonk_with_lookup_expression(num_vars),
//                     |_| ((), ()),
//                     |num_vars| {
//                         let (polys, challenges) = rand_vanilla_plonk_with_lookup_assignment(
//                             num_vars,
//                             seeded_std_rng(),
//                             seeded_std_rng(),
//                         );
//                         (polys, challenges, rand_vec(num_vars, seeded_std_rng()))
//                     },
//                 );
//             }
//         };
//     }

//     pub(super) use tests;
// }
