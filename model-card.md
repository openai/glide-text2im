# Overview

This card describes the diffusion model GLIDE (filtered) and noised CLIP model described in the paper [GLIDE: Towards
Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)

# Datasets

GLIDE (filtered) was trained on a filtered version of a dataset comprised of several hundred million text-image pairs
collected from the internet. We constructed a set of filters intended to remove all images of people, violent objects, and some
and hate symbols (see Appendix F of the paper for details). The size of the dataset after filtering was approximately
67M text-image pairs.

Our noised CLIP model which was trained on the dataset described above, augmented with a filtered version of the dataset used
to train the [original CLIP models](https://github.com/openai/clip). The total size of this augmented dataset is approximately 137M pairs.

# Performance

Qualitatively, we find that the generated images from GLIDE (filtered) often look semi-realistic, but the small size of the model hinders
its ability to bind attributes to objects and perform compositional tasks. Because the dataset used to train GLIDE
(filtered) has been preprocessed to remove images of people, this also limits its world knowledge, especially in regard
to concepts that involve people.
Finally, due to the dataset used to train GLIDE (filtered), the model has reduced capabilities to compose multiple objects in complex ways compared to models of a similar size trained on our internal dataset.

We do not directly measure quantitative metrics for GLIDE (filtered). In particular, most of the evaluations we report for our other models are biased against GLIDE (filtered), since they use prompts that often require generations of people. Evaluating people-free models remains an open area of research.

# Intended Use

We release these models to help advance research in generative modeling. Due to the limitations and biases of GLIDE (filtered), we do not currently recommend it for commercial use.

Functionally, these models are intended to be able to perform the following tasks for research purposes:
 * Generate images from natural language prompts
 * Iteratively edit and refine images using inpainting

These models are explicitly not intended to generate images of people or other subjects we filtered for (see Appendix F of the paper for details).

# Limitations

Despite the dataset filtering applied before training, GLIDE (filtered) continues to exhibit biases that extend beyond those found in images of people.
We explore some of these biases in our paper. For example:

  * It produces different outputs when asked to generate toys for boys and toys for girls.
  * It gravitates toward generating images of churches when asked to generate "a religious place",
    and this bias is amplified by classifier-free guidance.
  * It may have a greater propensity for generating hate symbols other than swastikas and confederate flags. Our filter
    for hate symbols focused specifically on these two cases, as we found few relevant images of hate symbols in our
    dataset. However, we also found that the model has diminished capabilities across a wider set of symbols.

GLIDE (filtered) can fail to produce realistic outputs for complex prompts or for prompts that involve concepts that are
not well-represented in its training data. While the data for the model was filtered to remove certain types of images,
the data still exhibits biases toward Western-centric concepts.
