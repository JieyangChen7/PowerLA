#! /bin/bash
set -x
set -e

src_dir=.
build_dir=build

mkdir -p $build_dir

cmake -S $src_dir -B $build_dir
cmake --build $build_dir -j 8