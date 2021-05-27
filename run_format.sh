#!/bin/bash
for f in ./*pp; do
    echo "file name is " $f
    clang-format  -i -style=file $f
done
