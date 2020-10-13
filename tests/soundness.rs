#[forbid(unsafe_code)]
use dync::*;

#[repr(align(256))]
#[derive(Copy, Clone)]
struct LargeAlign(u8);

// Test the soundness regression found in https://github.com/elrnv/dync/issues/4
#[test]
fn alignment() {
    let mut x: VecCopy = VecCopy::with_type::<LargeAlign>();
    x.push_as::<LargeAlign>(LargeAlign(0));

    let _ref_to_element = x.get_ref_as::<LargeAlign>(0).unwrap();
}
