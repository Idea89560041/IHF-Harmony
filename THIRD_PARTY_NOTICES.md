# Third-Party Notices

This repository is a project-specific implementation for IHF-Harmony. It keeps
the relationship to the source implementation explicit.

## Related Method

This implementation is based on:

- Weichen Fan, Jinghuan Chen, Ziwei Liu. "Hierarchy Flow For High-Fidelity
  Image-to-Image Translation." arXiv:2308.06909, 2023.

This codebase adapts the method from general CV image-to-image translation to
unpaired MRI harmonization, including a new training entry point, dataset
pipeline, artifact-aware normalization module, and manuscript-aligned
consistency losses.

## VGG Encoder Weights

`model/losses/vgg_model/vgg_normalised.pth` is used as a fixed VGG feature
encoder for perceptual/statistical loss computation. It is intentionally ignored
by git to keep the public repository small. Place the weight file at that path
before training or evaluation.
