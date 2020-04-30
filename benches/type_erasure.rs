use std::any::Any;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;

use data_buffer::{VecClone, VecCopy};

static SEED: [u8; 32] = [3; 32];

#[inline]
fn make_random_vec(n: usize) -> Vec<[f64; 3]> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n).map(move |_| [rng.gen::<f64>(); 3]).collect()
}

#[inline]
fn make_random_vec_any(n: usize) -> Vec<Box<dyn Any>> {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    (0..n)
        .map(move |_| {
            let b: Box<dyn Any> = Box::new([rng.gen::<f64>(); 3]);
            b
        })
        .collect()
}

#[inline]
fn make_random_vec_copy(n: usize) -> VecCopy {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let vec: Vec<_> = (0..n).map(move |_| [rng.gen::<f64>(); 3]).collect();
    vec.into()
}

#[inline]
fn make_random_vec_clone(n: usize) -> VecClone {
    let mut rng: StdRng = SeedableRng::from_seed(SEED);
    let vec: Vec<_> = (0..n).map(move |_| [rng.gen::<f64>(); 3]).collect();
    vec.into()
}

#[inline]
fn compute(x: f64, y: f64, z: f64) -> [f64; 3] {
    [
        x * 3.6321 + 42314.0 * y + z * 2.1,
        y * 3.6321 + 42314.0 * z + x * 2.1,
        z * 3.6321 + 42314.0 * x + y * 2.1,
    ]
}

#[inline]
fn vec_compute(v: &mut Vec<[f64; 3]>) {
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
        let a = (&mut *a).downcast_mut::<[f64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_copy_compute(v: &mut VecCopy) {
    for a in v.iter_value_mut() {
        let a = a.downcast::<[f64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

#[inline]
fn vec_clone_compute(v: &mut VecClone) {
    for a in v.iter_value_mut() {
        let a = a.downcast::<[f64; 3]>().unwrap();
        let res = compute(a[0], a[1], a[2]);
        a[0] = res[0];
        a[1] = res[1];
        a[2] = res[2];
    }
}

fn type_erasure(c: &mut Criterion) {
    let mut group = c.benchmark_group("Type Erasure");

    for &buf_size in &[3000, 30_000, 90_000, 180_000, 300_000, 600_000, 900_000] {
        group.bench_function(BenchmarkId::new("Vec<f64>", buf_size), |b| {
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

        group.bench_function(BenchmarkId::new("VecClone", buf_size), |b| {
            let mut v = make_random_vec_clone(buf_size);
            b.iter(|| {
                vec_clone_compute(&mut v);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, type_erasure);
criterion_main!(benches);
