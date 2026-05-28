# Environment Setup


## Step 1. Create conda environment and activate

```bash
conda create --name maptracker python=3.8 -y
conda activate maptracker
```

## Step 2. Install PyTorch

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## Step 3. Install MMCV series

```bash
# Force the mmcv 2.1.0 source build to use a C++17-capable toolchain.
# Without these, the build fails on hosts whose default gcc still
# defaults to C++14.
export CXXFLAGS="-std=c++17"
export CFLAGS="-std=c++17"

pip install mmcv==2.1.0  # if pip cached an older wheel: pip install --no-cache-dir --force-reinstall mmcv==2.1.0
pip install mmengine==0.10.5 mmdet==3.3.0 mmsegmentation==1.2.0
```

Then install `mmdetection3d` in editable mode. Run these commands **from
inside the maptracker repo root** so the directory tree matches what the
configs and `docs/data_preparation.md` expect:

```bash
# cwd: <maptracker repo root>
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.4.0
pip install -e .
cd ..  # back to maptracker repo root
```

## Step 4. Install other requirements

`requirements.txt` on this branch is modified from the original file and
intentionally does **not** pin mmcv / mmdet / mmengine / mmsegmentation /
mmdetection3d (those came from Step 3 in a specific order):

```bash
pip install -r requirements.txt
```

## Step 5. Verify the install

Sanity-check the environment before moving on to data preparation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.1.2+cu121 True

python -c "import mmcv, mmcv.ops, mmengine, mmdet, mmseg, mmdet3d; print('mm stack ok')"
# expected: mm stack ok

python -c "import plugin; print('plugin ok')"
# expected: plugin ok
```

If `torch.cuda.is_available()` returns `False`, your host driver is older
than 525.60.13 (see Prerequisites). If `import mmcv.ops` raises, the mmcv
CUDA extension did not build cleanly — re-run Step 3 with
`pip install --no-cache-dir --force-reinstall mmcv==2.1.0`.

## Alternative: frozen reference environment

If the step-by-step install above breaks (most often: mmcv build failure on
unusual gcc/CUDA combos), the repo ships a frozen pip-list snapshot of the
exact env we use:

```bash
conda env create -f environment.yml
conda activate maptracker
```

This pins every transitive package. Use it as a fallback or to debug a
discrepancy with the canonical install above.

## Next steps

Once the verify commands pass, continue with
[`docs/data_preparation.md`](data_preparation.md) for dataset setup.
