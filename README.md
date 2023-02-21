# CloZ: Natural Language Guided Clothing Design System 

## Abstract

**CloZ** is a clothing design system facilitated by natural language prompts. Cloz supports two main functions; **1) generating clothing images via natural language prompting. 2) editing generated images by replacing keywords from previous prompts.** 

Inspired by [FACAD [1]](https://github.com/xuewyang/Fashion_Captioning), we first built the [nordstrom96568 dataset](https://huggingface.co/datasets/jasonchoi3/nordstrom96568), which consists of 96568 (prompt, clothing image) pairs. Then we trained [stable diffusion [2] 2.1](https://github.com/Stability-AI/stablediffusion) with our dataset to generate clothing images using prompts. The editing function was implemented by [CycleDiffusion [3]](https://github.com/ChenWu98/cycle-diffusion). Also, we designed the CloZ's web-based interface based on guidelines of prior research [4]. 

To the best of our knowledge, CloZ is the first clothing design system using natural language guidance.

## Requirements

```
# Please setup CUDA, torch first. 

pip install requirements.txt
```


## Development

TBA

## References

[1] Yang, X., Zhang, H., Jin, D., Liu, Y., Wu, C. H., Tan, J., ... & Wang, X. (2020). Fashion captioning: Towards generating accurate descriptions with semantic rewards. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIII 16 (pp. 1-17). Springer International Publishing.

[2] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).


[3] Wu, C. H., & De la Torre, F. (2022). Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance. arXiv preprint arXiv:2210.05559.

[4] Ko, H. K., Park, G., Jeon, H., Jo, J., Kim, J., & Seo, J. (2022). Large-scale Text-to-Image Generation Models for Visual Artists' Creative Works. arXiv preprint arXiv:2210.08477.
