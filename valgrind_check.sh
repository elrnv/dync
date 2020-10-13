#!/bin/sh
cargo test 2>&1 | grep "Running.*target/debug/deps/.*" | sed 's/.*Running //g' | xargs -I{} valgrind --error-exitcode=1 --leak-check=full --show-leak-kinds=all ./{}
