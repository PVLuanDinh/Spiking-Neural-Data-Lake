"""DEV extraction: per session, Natural Movie One responses as (cells, repeats, segments) of mean dF/F, plus running
speed and pupil area on the same grid. Range-reads only the slices needed; writes one small npz per session."""
import json
import sys
import urllib.request

import h5py
import numpy as np

from rangefile import RangeFile

SEG = 30  # movie frames per segment: 30 frames = 1 s at 30 Hz, so 30 segments per 900-frame repeat


def nwb_link(exp_id):
    q = ("https://api.brain-map.org/api/v2/data/query.json?criteria=model::WellKnownFile,rma::criteria,"
         f"well_known_file_type[id%24eq514173063],[attachable_id%24eq{exp_id}]")
    w = json.load(urllib.request.urlopen(q, timeout=120))["msg"]
    assert len(w) == 1, (exp_id, w)
    return "https://api.brain-map.org" + w[0]["download_link"]


def windows(stim_ts, stim_frames):
    """(repeat, segment) -> (t_start, t_end) of each 30-frame movie segment."""
    starts = np.flatnonzero(stim_frames == 0)
    out = []
    for s in starts:
        rep = []
        for k in range(900 // SEG):
            i0, i1 = s + k * SEG, s + (k + 1) * SEG - 1
            rep.append((stim_ts[i0], stim_ts[i1] + 1 / 30))
        out.append(rep)
    return np.array(out)  # (repeats, segments, 2)


def bin_mean(t, x, win):
    """Mean of samples x at times t inside each window; x may be (n, T) or (T,)."""
    lo = np.searchsorted(t, win[..., 0])
    hi = np.searchsorted(t, win[..., 1])
    if x.ndim == 1:
        return np.array([[np.nanmean(x[a:b]) if b > a else np.nan for a, b in zip(r0, r1)] for r0, r1 in zip(lo, hi)])
    return np.stack([np.stack([x[:, a:b].mean(axis=1) for a, b in zip(r0, r1)], axis=1) for r0, r1 in zip(lo, hi)], axis=1)


def extract(exp_id, eye_key, out):
    f = RangeFile(nwb_link(exp_id))
    h = h5py.File(f, "r")
    p = "processing/brain_observatory_pipeline/"
    st = h["stimulus/presentation/natural_movie_one_stimulus"]
    stim_ts, stim_fr = st["timestamps"][:], st["data"][:]
    win = windows(stim_ts, stim_fr)
    t2p = h[p + "DfOverF/imaging_plane_1/timestamps"][:]
    i0 = max(0, int(np.searchsorted(t2p, win.min())) - 2)
    i1 = int(np.searchsorted(t2p, win.max())) + 2
    dff = h[p + "DfOverF/imaging_plane_1/data"][:, i0:i1]
    resp = bin_mean(t2p[i0:i1], dff, win)  # (cells, repeats, segments)
    cells = h[p + "ImageSegmentation/cell_specimen_ids"][:]
    rs_t = h[p + "BehavioralTimeSeries/running_speed/timestamps"][:]
    rs = h[p + "BehavioralTimeSeries/running_speed/data"][:]
    run = bin_mean(rs_t, rs, win)
    meta = {k: h["general/" + k][()].decode() for k in ("session_type", "targeted_structure", "experiment_container_id")}
    nwb_mb = f.fetched / 1e6
    e = RangeFile("https://allen-brain-observatory.s3.amazonaws.com/" + eye_key)
    eh = h5py.File(e, "r")
    et = eh["synced_frame_timestamps/values"][:]
    pa = eh["new_pupil_areas/values"][:]
    pupil = bin_mean(et, pa, win)
    np.savez_compressed(out, resp=resp.astype(np.float32), cells=cells, run=run, pupil=pupil, win=win, **meta)
    return resp.shape, meta, round(nwb_mb, 1), round(e.fetched / 1e6, 1), float(np.mean(np.isnan(pupil)))


if __name__ == "__main__":
    exp_id, eye_key, out = sys.argv[1], sys.argv[2], sys.argv[3]
    print(exp_id, *extract(exp_id, eye_key, out))
