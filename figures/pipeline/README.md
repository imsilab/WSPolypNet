# Fourteen-stage pipeline visualization

1. Full ROI/model input
2. Top-left 144×144 crop resized to 224×224
3. Top-right crop
4. Bottom-left crop
5. Bottom-right crop
6. Full-view CAM
7. Top-left crop CAM
8. Top-right crop CAM
9. Bottom-left crop CAM
10. Bottom-right crop CAM
11. Pixel-wise maximum-fused five-view CAM
12. Selected seed-frame fused CAM
13. NMS candidate points and selected point
14. Final MedSAM2 bounding boxes over all 30 frames
