use std::any::Any;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use dync::{dync_trait, into_dyn, traits::HasDrop, VecCopy, VecDrop};

static SEED: [u8; 32] = [3; 32];

#[dync_trait]
pub trait DynClone {}
impl DynClone for [i64; 3] {}

#[inline]
fn make_random_vec(n: usize) -> Vec<[i64; 3]> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| [rng.gen::<i64>(); 3]).collect()
}

#[inline]
fn make_random_vec_any(n: usize) -> Vec<Box<dyn Any>> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n)
        .map(move |_| {
            let b: Box<dyn Any> = Box::new([rng.gen::<i64>(); 3]);
            b
        })
        .collect()
}

#[inline]
fn make_random_vec_copy(n: usize) -> VecCopy {
    make_random_vec(n).into()
}

#[inline]
fn make_random_vec_drop(n: usize) -> VecDrop<DynCloneVTable> {
    make_random_vec(n).into()
}

#[inline]
fn make_random_vec_dyn(n: usize) -> VecCopy<dyn Any> {
    let vec: VecCopy = make_random_vec_copy(n);
    into_dyn![VecCopy<dyn Any>](vec)
}

#[inline]
fn compute(x: i64, y: i64, z: i64) -> [i64; 3] {
    [
        x * 3 - 5 * y + z * 2,
        y * 3 - 5 * z + x * 2,
        z * 3 - 5 * x + y * 2,
    ]
}

#[inline]
fn vec_compute(v: &mut Vec<[i64; 3]>) {
    for a in v.iter_mut() {
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_any_compute(v: &mut Vec<Box<dyn Any>>) {
    for ref mut a in v.iter_mut() {
        let a = (&mut *a).downcast_mut::<[i64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_copy_compute<V>(v: &mut VecCopy<V>) {
    for a in v.iter_mut() {
        let a = a.downcast::<[i64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_drop_compute<V: Clone + HasDrop>(v: &mut VecDrop<V>) {
    for a in v.iter_mut() {
        let a = a.downcast::<[i64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_dyn_compute(v: &mut VecCopy<dyn Any>) {
    for a in v.iter_mut() {
        let a = a.downcast::<[i64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

fn type_erasure(c: &mut Criterion) {
    let mut group = c.benchmark_group("Type Erasure");

    for &buf_size in &[3000, 30_000, 90_000, 180_000, 300_000, 600_000, 900_000] {
        group.bench_function(BenchmarkId::new("Vec<[i64;3]>", buf_size), |b| {
            let mut v = make_random_vec(buf_size);
            b.iter(|| {
                vec_compute(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("Vec<Box<dyn Any>>", buf_size), |b| {
            let mut v = make_random_vec_any(buf_size);
            b.iter(|| {
                vec_any_compute(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("VecCopy", buf_size), |b| {
            let mut v = make_random_vec_copy(buf_size);
            b.iter(|| {
                vec_copy_compute(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("VecDrop", buf_size), |b| {
            let mut v = make_random_vec_drop(buf_size);
            b.iter(|| {
                vec_drop_compute(&mut v);
            })
        });

        group.bench_function(BenchmarkId::new("VecDyn", buf_size), |b| {
            let mut v = make_random_vec_dyn(buf_size);
            b.iter(|| {
                vec_dyn_compute(&mut v);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, type_erasure);
criterion_main!(benches);
