import warnings

import pytest

from aaa._compat import deprecated


def test_deprecated_warns_with_replacement_and_still_works():
    @deprecated(since="0.3", removed_in="0.5", use="new_name")
    def old(x):
        return x + 1

    with pytest.warns(
        DeprecationWarning, match=r"old.*since 0\.3.*removed in 0\.5.*use new_name"
    ) as rec:
        assert old(1) == 2
    assert rec[0].filename == __file__  # attributed to the caller, not to the helper
    assert old.__deprecated__.startswith("test_deprecated")


def test_methods_and_no_replacement():
    class A:
        @deprecated(since="0.3", removed_in="0.5")
        def m(self):
            return "ok"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert A().m() == "ok"
    assert "use" not in str(w[0].message)
