"""Brain-free separating witness on the existing zero-growth Part D path.

This is a diagnostic oracle, not learned performance and not an Engine result.
Run with the project's free-threaded interpreter and pinned source on PYTHONPATH.
"""
import hashlib
import inspect
import json

import numpy as np

from snn_data_lake import require_free_threaded
from snn_data_lake.instruments import inseparable_mapping as im
from snn_data_lake.instruments.g5_part_d_cells import _learner
from snn_data_lake.instruments.g5_readout_cells import pd_pool, pool_features
from snn_data_lake.readout import _IXN_INTENTS

require_free_threaded("pool separator audit", strict=True)
pool = pd_pool(9900, "carry")
learner = _learner(list(_IXN_INTENTS), 256, 9900)
features = []
for uv in im.UV:
    pool.reset_carriers()
    features.append(pool_features(pool, im.TEXTS[uv]).tolist())
phis = np.asarray([
    [learner._psp(learner.t_post - rank) for rank in learner.encode(x)]
    for x in features
])
labels = [im.label_xor(*uv) for uv in im.UV]
y = np.zeros((4, len(learner.responses)))
for i, label in enumerate(labels):
    y[i, learner.responses.index(label)] = 1
w = np.linalg.lstsq(phis, y, rcond=None)[0].T
for neuron, row in zip(learner.neurons, w, strict=True):
    neuron.w = row.tolist()
answers = [learner.respond(x) for x in features]
result = {
    "seed": 9900,
    "scope": "constructed, brain-free separating diagnostic; no learning, no lake, no full-Engine result",
    "unique_features": len({tuple(x) for x in features}),
    "readout_psp_rank": int(np.linalg.matrix_rank(phis)),
    "answers": answers,
    "labels": labels,
    "real_respond_separator_verified": answers == labels,
    "matrix_residual_max": float(np.abs(phis @ w.T - y).max()),
    "source_files": {
        inspect.getsourcefile(obj): hashlib.sha256(open(inspect.getsourcefile(obj), "rb").read()).hexdigest()
        for obj in (pd_pool, pool_features, _learner, type(learner))
    },
}
print(json.dumps(result, sort_keys=True))
