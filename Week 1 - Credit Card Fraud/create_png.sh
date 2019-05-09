#!/bin/bash
find . -name '*.dot' | xargs -I{files} dot -Tpng {files} -O
