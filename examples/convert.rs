//! Given that one trait inherits the behaviour of another, it is possible to downcast a value of a
//! more specific trait to a more general one.
//!
//! For this to work, we must use the `#[dync_mod]` attribute on a local module that defines the
//! traits in question.

use dync::BoxValue;
use dync_derive::dync_mod;
use std::rc::Rc;

#[dync_mod]
mod traits {
    pub trait ValBase: Clone + PartialEq + std::fmt::Debug {}
    impl<T> ValBase for T where T: Clone + PartialEq + std::fmt::Debug {}

    // Overlaps in the referenced traits are resolved.
    pub trait Val: ValBase + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}
    impl<T> Val for T where T: ValBase + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {}
}

use crate::traits::*;

type MyVal = BoxValue<ValVTable>;
type MyValBase = BoxValue<ValBaseVTable>;

fn main() {
    let v = MyVal::new(Rc::new(32i32));
    let u: MyValBase = v.upcast();
    let w = u.clone();
    println!("w is {:?}", &w);
    let w = w.downcast::<Rc<i32>>().unwrap();
    let count = Rc::strong_count(&w);
    println!("... with reference count {}", count);
}
