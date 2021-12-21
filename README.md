# GLIDE

This is the official codebase for running the small, filtered-data GLIDE model from [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).

For details on the pre-trained models in this repository, see the [Model Card](model-card.md).

# Usage

To install this package, clone this repository and then run:

```
pip install -e .
```

For detailed usage examples, see the [notebooks](notebooks) directory.

 * The [text2im](notebooks/text2im.ipynb) notebook shows how to use GLIDE (filtered) with classifier-free guidance to produce images conditioned on text prompts.
 * The [inpaint](notebooks/inpaint.ipynb) notebook shows how to use GLIDE (filtered) to fill in a masked region of an image, conditioned on a text prompt.
 * The [clip_guided](notebooks/clip_guided.ipynb) notebook shows how to use GLIDE (filtered) + a filtered noise-aware CLIP model to produce images conditioned on text prompts.