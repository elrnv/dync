//! A common technique for improving the performance of a program that evaluates a function for the
//! same inputs many times is caching.
//!
//! For instance suppose we have a costly function that computes a unique value for a given
//! integer many times. Furthermore assume that the integer inputs to this program repeat
//! over time, which makes it cheaper to pull the result from a table of previously computed
//! values.

use dync::{BoxValue, Value, VecDyn};
use dync_derive::dync_trait;
use std::collections::HashMap;
use std::time::Instant;
use std::rc::Rc;

use rand::prelude::*;
use rand::distributions::Alphanumeric;

// Define a trait for a value that can be stored in a HashTable.
// This is quite tricky to do with trait objects. For instance, see the following post for how to
// make a trait object work with the Eq trait (which is required for HashMap values):
// https://stackoverflow.com/questions/25339603/how-to-test-for-equality-between-trait-objects
#[dync_trait]
trait HTValue: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}
impl<T> HTValue for T where T: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}

// An alias for a VecDyn of HTValue types.
type VecDynCached = VecDyn<HTValueVTable>;

// A sample function to produce 3 large unsigned integer values given some small seed.
fn costly_computation_int(seed: u8) -> [u128; 3] {
    let mut rng: StdRng = SeedableRng::from_seed([seed; 32]);
    [rng.gen(), rng.gen(), rng.gen()]
}

// Another sample computation that produces random strings.
fn costly_computation_str(seed: u8) -> String {
    let rng: StdRng = SeedableRng::from_seed([seed; 32]);
    rng.sample_iter(&Alphanumeric)
        .take(30)
        .collect()
}

fn main() {
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    {
        let mut int_values = VecDynCached::with_type::<[u128; 3]>();
        let mut str_values = VecDynCached::with_type::<String>();

        let start_time = Instant::now();
        for _ in 0..50_000 {
            // Generate a random seed.
            let seed = rng.gen::<u8>();

            int_values.push(costly_computation_int(seed));
            str_values.push(costly_computation_str(seed));
        }
        println!("non cached loop: {} milliseconds", start_time.elapsed().as_millis());
    }

    {
        let mut cache = HashMap::<u8, BoxValue<HTValueVTable>>::new();

        let mut int_values = VecDynCached::with_type::<Rc<[u128; 3]>>();
        let mut str_values = VecDynCached::with_type::<Rc<String>>();

        let start_time = Instant::now();
        for _ in 0..50_000 {
            // Generate a random seed.
            let seed = rng.gen::<u8>();

            let int_value = cache.entry(seed).or_insert_with(|| Value::new(Rc::new(costly_computation_int(seed))));
            int_values.push_cloned(int_value.as_ref());

            let str_value = cache.entry(seed).or_insert_with(|| Value::new(Rc::new(costly_computation_int(seed))));
            str_values.push_cloned(str_value.as_ref());
        }
        println!("cached loop: {} milliseconds", start_time.elapsed().as_millis());
    }
}
