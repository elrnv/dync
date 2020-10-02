use dync::*;

#[repr(align(256))]
#[derive(Copy, Clone)]
struct LargeAlign(u8);

impl VTable<LargeAlign> for LargeAlign {
    fn build_vtable() -> Self {
        LargeAlign(0)
    }
}

#[test]
fn alignment() {
    // The backing storage for a VecCopy is a u8, meaning that casting to a type
    // with different alignment triggers undefined behavior.
    // https://github.com/elrnv/dync/blob/c133056676582dd0e28c14526175d0c9ae01a905/src/vec_copy.rs#L64-L65
    let mut x = VecCopy::<LargeAlign>::with_type();
    x.push_as::<LargeAlign>(LargeAlign(0));

    let _ref_to_element = x.get_ref_as::<LargeAlign>(0).unwrap();
}
