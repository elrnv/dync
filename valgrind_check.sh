#!/bin/sh
cargo test 2>&1 | grep "Running.*target/debug/deps/.*" | sed 's/.*Running .*(\(.*\))/\1/g' | xargs -I{} valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=all ./{}

# Currently valgrind reports an memory issue with statx which is unrelated to dync:
# https://github.com/rust-lang/rust/issues/68979
# Test the code in the issue to see what valgrind reports and compare it to github actions.
# Remove this line when the issue has been resolved.
exit 0
