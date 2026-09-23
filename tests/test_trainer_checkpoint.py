"""T101 panel G2: a legacy snntorch SpikingReadout checkpoint is refused by name."""

import io

import pytest
import torch

from snn_data_lake.training.trainer import SpikingReadout, load_checkpoint, save_checkpoint

LEGACY_KEYS = ("lif.graded_spikes_factor", "lif.reset_mechanism_val")


def _payload(extra_keys=()):
    stream = io.BytesIO()
    save_checkpoint(SpikingReadout(4, 2), stream)
    stream.seek(0)
    ckpt = torch.load(stream, map_location="cpu")
    for key in extra_keys:
        ckpt["state_dict"][key] = torch.tensor(1.0)
    out = io.BytesIO()
    torch.save(ckpt, out)
    out.seek(0)
    return out


def test_current_checkpoint_round_trips():
    assert isinstance(load_checkpoint(_payload()), SpikingReadout)


@pytest.mark.parametrize("keys", [LEGACY_KEYS, LEGACY_KEYS[:1], LEGACY_KEYS[1:]])
def test_legacy_snntorch_keys_are_refused_by_name(keys):
    with pytest.raises(ValueError, match="Legacy snntorch SpikingReadout checkpoint"):
        load_checkpoint(_payload(keys))


def test_an_unrelated_extra_key_stays_strict():
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_checkpoint(_payload(("lif.unrelated",)))
