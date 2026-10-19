"""The hatch build hook IS `infra.native_build` (Codex B2 integrator MAJOR 1:
`snn_data_lake._native` became a hard runtime dependency of the PFC, yet `pip install .` never
built it and the extension is gitignored, so an ordinary install imported nothing). pyproject's
custom wheel hook must name that module (ONE build path, P2 -- the hook does not fork
native_build's logic), the hook body must build EVERY time (an existing extension may be stale),
and the wheel must stop claiming `py3-none-any` once it carries a compiled extension (`infer_tag`
+ non-pure: a .pyd in a pure wheel installs anywhere and imports nowhere else).

End-to-end red/green (a temp-target `pip install` + PFC import) is a lane log, not a unit test: it
needs the Rust toolchain and ~12 s of maturin. Run: pytest tests/test_hatch_build.py
"""

import pathlib
import sysconfig
import tomllib

from snn_data_lake.infra import native_build as nb

PYPROJECT = tomllib.loads((nb.ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WHEEL = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]


def test_pyproject_wheel_hook_is_native_build():
    hook_path = (nb.ROOT / WHEEL["hooks"]["custom"]["path"]).resolve()
    assert hook_path == pathlib.Path(nb.__file__).resolve(), hook_path
    assert callable(nb.get_build_hook)
    assert "maturin>=1.15,<2" in PYPROJECT["build-system"]["requires"]
    assert "src/snn_data_lake/_native.*" in WHEEL["artifacts"]  # gitignored, shipped anyway


def test_hook_rebuilds_even_when_an_extension_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "PKG", tmp_path)
    calls, data = [], {}
    nb.hook_initialize(data, build=calls.append)
    assert calls == [[]] and data == {"pure_python": False, "infer_tag": True}
    # THIS interpreter's extension, present and possibly stale, proves nothing about its Rust source
    (tmp_path / ("_native" + sysconfig.get_config_var("EXT_SUFFIX"))).touch()
    nb.hook_initialize(data, build=calls.append)
    assert calls == [[], []], "skipped the build because an extension file exists (stale ship)"
