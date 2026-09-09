"""Minimal test harness for the Mojo 1.0.0 toolchain.

This Mojo build ships no `mojo test` runner and no `testing` module, so
every test file is a standalone executable:

    mojo run -I src -I tests tests/test_<area>.mojo

Each test file defines `def test_*() raises` functions that use `expect()`
below, and a `main()` that runs them, collecting failures.  A non-zero
exit code signals failure (the last `raise` from main propagates).
"""


def expect(cond: Bool, msg: String) raises:
    """Assert `cond`, raising Error with `msg` otherwise."""
    if not cond:
        raise Error("expect failed: " + msg)


def expect_eq(a: Int, b: Int, msg: String) raises:
    """Assert Int equality with a descriptive message."""
    if a != b:
        raise Error(
            msg + " (expected " + String(b) + ", got " + String(a) + ")"
        )


def expect_str(a: String, b: String, msg: String) raises:
    """Assert String equality with a descriptive message."""
    if a != b:
        raise Error(msg + " (expected " + repr(b) + ", got " + repr(a) + ")")


def summarize(failures: List[String]) raises:
    """Print failure summary and raise if any test failed."""
    if len(failures) > 0:
        print()
        print(String(len(failures)) + " test(s) FAILED:")
        for f in failures:
            print("  - " + f)
        raise Error(String(len(failures)) + " test(s) failed")
    print()
    print("ALL TESTS PASSED")
