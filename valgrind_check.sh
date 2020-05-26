#!/bin/sh

export bin=$(cargo test 2>&1 | grep "Running.*target/debug/deps/.*" | sed 's/.*Running //g')
valgrind --error-exitcode=1 --leak-check=full ./$bin
