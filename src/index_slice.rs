use std::ops;

/// A helper trait to scale a standard Rust range by a given scale.
///
/// This allows us to use all of Rust's range types for indexing out `Slice`s.
pub trait ScaleRange {
    fn scale_range(self, scale: usize) -> Self;
}

impl ScaleRange for ops::Range<usize> {
    #[inline]
    fn scale_range(mut self, scale: usize) -> Self {
        self.start *= scale;
        self.end *= scale;
        self
    }
}

impl ScaleRange for ops::RangeTo<usize> {
    #[inline]
    fn scale_range(mut self, scale: usize) -> Self {
        self.end *= scale;
        self
    }
}

impl ScaleRange for ops::RangeFrom<usize> {
    #[inline]
    fn scale_range(mut self, scale: usize) -> Self {
        self.start *= scale;
        self
    }
}

impl ScaleRange for ops::RangeFull {
    #[inline]
    fn scale_range(self, _: usize) -> Self {
        // Scaling an entire range does nothing.
        self
    }
}

impl ScaleRange for ops::RangeInclusive<usize> {
    #[inline]
    fn scale_range(self, scale: usize) -> Self {
        self.start() * scale..=self.end() * scale
    }
}

impl ScaleRange for ops::RangeToInclusive<usize> {
    #[inline]
    fn scale_range(mut self, scale: usize) -> Self {
        self.end *= scale;
        self
    }
}
