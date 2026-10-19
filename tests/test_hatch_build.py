"""The hatch build hook IS `infra.native_build` (Codex B2 integrator MAJOR 1:
`snn_data_lake._native` became a hard runtime dependency of the PFC, yet `pip install .` never
built it and the extension is gitignored, so an ordinary install imported nothing). pyproject's
custom wheel hook must name that module (ONE build path, P2 -- the hook does not fork
native_build's logic), the hook body must build only when THIS interpreter's extension is absent,
and the wheel must stop claiming `py3-none-any` once it carries a compiled extension (`infer_tag`
+ non-pure: a .pyd in a pure wheel installs anywhere and imports nowhere else).

End-to-end red/green (a temp-target `pip install` + PFC import) is a lane log, not a unit test: it
needs the Rust toolchain and ~12 s of maturin. Run: pytest tests/test_hatch_build.py
"""

import pathlib
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


def test_hook_builds_only_when_this_interpreters_extension_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(nb, "PKG", tmp_path)
    calls, data = [], {}
    nb.hook_initialize(data, build=calls.append)
    assert calls == [[]] and data == {"pure_python": False, "infer_tag": True}
    nb.placed().touch()  # another interpreter's _native.* would NOT count: the suffix is the key
    nb.hook_initialize(data, build=calls.append)
    assert calls == [[]], "rebuilt although this interpreter's extension is present"
