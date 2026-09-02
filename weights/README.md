# Released checkpoints

The selected checkpoints live in each model's `Checkpoints/` directory so the original command-line defaults continue to work.

| Backbone | Selected checkpoint | Selection criterion |
|---|---|---|
| Slow-R50 | `Slow-R50/Checkpoints/epoch_020.pth` | validation CorLoc@0.5 |
| SlowFast-R50 | `SlowFast-R50/Checkpoints/epoch_044.pth` | validation CorLoc@0.5 |
| R3D-18 | `R3D-18/Checkpoints/epoch_004.pth` | validation CorLoc@0.5 |
| R(2+1)D-18 | `R(2+1)D-18/Checkpoints/epoch_009.pth` | validation CorLoc@0.5 |
| X3D-M | `X3D/Checkpoints/epoch_014.pth` | validation CorLoc@0.5; adopted pipeline backbone |

These are inference-only public exports containing the model state, epoch, evaluation metrics, and public ROI-manifest digest. Optimizer, scheduler, gradient-scaler, local paths, and training arguments were removed.

Files larger than GitHub's regular file limit are tracked with Git LFS. Run `git lfs pull` after cloning.

## SHA-256

```text
b8b2e1c46b42c45f48010bfdeee621d02f9929ad18c92351a115389bf7b192f9  Slow-R50/Checkpoints/epoch_020.pth
e5c1b35b49e1eda40ac75f65fd496d3c0c4796c8396a1784ef2f013932b7e9c8  SlowFast-R50/Checkpoints/epoch_044.pth
0f0e1a774882a785a5a2ca474321ff60828f89ff580f95c2c4e2961dd714e7ba  R3D-18/Checkpoints/epoch_004.pth
a84ded3812aef4b5bc09092a04f1d18311ad48b48ad3dad265988b0bed7b4753  R(2+1)D-18/Checkpoints/epoch_009.pth
6544a4e77050b0e3cc8f6a11c8893cd397e0a034c6a361af41b5fe86b9a69e8f  X3D/Checkpoints/epoch_014.pth
```

MedSAM2 weights are not redistributed; see `external/MedSAM2/README.md`.
