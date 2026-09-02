# Dataset preparation

Datasets are intentionally excluded from this repository.

Training data use one MP4 per sample under numeric folders. The folder number is used to select a shared FOV entry from `roi_manifest.json` under either the `negative/<folder>` or `positive/<folder>` namespace.

Validation data use this layout:

```text
ValidationData/
├── video/<numeric-folder>/*.mp4
└── annotation/<numeric-folder>/*.txt
```

Within each folder, videos and annotation files are naturally sorted. Annotation files are consumed sequentially according to each video's decoded frame count. Each text file contains:

```text
<number of boxes>
x1 y1 x2 y2
...
```

Coordinates are in the original decoded-frame coordinate system. The loader maps predictions between the original frame and the deterministic evaluation ROI canvas.

The included `roi_manifest.json` contains geometry only. Its dataset roots are relative placeholders and may be changed when rebuilding the manifest for another dataset copy.
