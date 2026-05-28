# tests/

These are dataset / rendering smoke tests, not a pytest suite. They are
intended to be run manually after preparing the relevant datasets locally.

Most tests assume the following layout (relative to the repo root):

- `./datasets/RDX/server_root/...` — read-only mount of the HRDX raw data
  server (see `docs/data_preparation.md` once HRDX is released).
- `./datasets/RDX/HRDX_annotations/` — annotation root used by individual tests.
- `./datasets/RDX/*.pkl` — annotation pickles built by
  `tools/data_converter/generate_rdx_pkl.py`.
- `./datasets/nuscenes/...` — for `test_nusc_dataset.py`.

Adjust the `data_root`, `data_server_mountpoint`, `ann_file`, and `out_dir`
arguments at the bottom of each script to match your local layout, or set
the `RDX_*` and `NUSCENES_ROOT` environment variables that the tools under
`tools/` already honor. None of these scripts mutate datasets in-place.
