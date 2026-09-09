#!/usr/bin/env bash
# Run every test file under tests/ with the shared include paths.
# Exits non-zero on the first failing file.  Each test file is a
# standalone Mojo program (this Mojo build has no `mojo test` runner).
set -u
cd "$(dirname "$0")/.." || exit 1

MOJO="${MOJO:-mojo}"
failed=0
for f in tests/test_*.mojo; do
    echo "== $f =="
    "$MOJO" run -I src -I tests "$f" || {
        echo "FAILED: $f"
        failed=1
    }
    echo
done

if [ "$failed" -ne 0 ]; then
    echo "SOME TEST FILES FAILED"
    exit 1
fi
echo "ALL TEST FILES PASSED"
