# Full Engine runtime recovery

Recovered in an isolated directory outside the repository. No repository source, safety-envelope file, policy certification, or live lake was modified by this runtime work.

- CPython 3.14.6 free-threaded; Torch 2.15.0 CPU; NumPy 2.5.4; strict free-threaded Brain import passed.
- Core packages + test/build tools: exact wheel versions/hashes in `core-lock.json` (31 packages); pip receipt `core-install.json`.
- Rust 1.96.0 installed with isolated CARGO_HOME and RUSTUP_HOME under the visualization workspace's `engine-runtime`. Native kernel built by the incumbent `snn_data_lake.infra.native_build` on a full source snapshot at commit c4c10b285f1741aa798d5ee9c5296ae371fabecf.
- Native wheel: `sndl_native-0.1.0-cp314-cp314t-macosx_11_0_arm64.whl`; SHA256 fbb553c6b25b4e84a0ee54185a02af8d84601eca790e9c63305616d2947b4feb. Native extension is present only in the external snapshot's package, not the main repo/src.
- Full `engine.brain` import now succeeds with the GIL off. Original import failure was missing mujoco; `_native`, scipy, pyarrow, psutil, Brian2 and pytest were also absent before recovery.
- Baseline tests `tests/test_g9_posed_actor.py` and `tests/test_g9_growth_adapter.py`: 31 passed in 50.07 seconds, one serial worker, 600-second child deadline. These are existing software controls; they do not demonstrate the blocked deciding growth campaign.

## Running modified Python source

`run-engine.sh` defaults to the immutable `engine-runtime/baseline-source`. A separate `engine-runtime/working-source` copy includes the same compiled native extension. Copy only the changed src/tests files from the working repository into that snapshot, leaving the baseline intact; then set `ENGINE_SOURCE_ROOT` to `working-source` before invoking the wrapper. Main source shadows a separately installed package, so the native wheel installed in a second site alone would not make `_native` available under main repo/src. Do not use `maturin develop` or an editable .pth as a shortcut.

For tests, run from the chosen snapshot root with a bounded subprocess controller. Set NORD_HOME to an isolated temporary directory and SNDL_R2_CHILD_DEADLINE_NS to time.time_ns()+budget_s*10**9 at launch. The tests' conftest provides per-test and per-module isolated homes and respects a shorter caller deadline. Use `-p no:cacheprovider --no-cov`, no xdist/parallel Brain workers initially. Example test arguments: `-m pytest -p no:cacheprovider --no-cov -x -vv tests/test_g9_posed_actor.py tests/test_g9_growth_adapter.py`.

Brian2 2.10.1 is installed but reference simulations must remain in their existing isolated subprocess path because its import enables the GIL. No live home, daemon or physical arm path was exercised.
