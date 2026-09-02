# MedSAM2 dependency

MedSAM2 is not redistributed in this repository. Install it at this exact location:

```bash
git clone https://github.com/bowang-lab/MedSAM2.git external/MedSAM2
mkdir -p external/MedSAM2/checkpoints
wget -O external/MedSAM2/checkpoints/MedSAM2_latest.pt \
  https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
```

Expected checkpoint:

- Path: `external/MedSAM2/checkpoints/MedSAM2_latest.pt`
- SHA-256: `c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060`
- Configuration used by the pipeline: `configs/sam2.1_hiera_t512.yaml`

MedSAM2 remains subject to its upstream license and terms.
