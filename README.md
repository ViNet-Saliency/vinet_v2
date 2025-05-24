# ViNet++: Minimalistic Video Saliency Prediction via Efficient Decoder & Spatio Temporal Action Cues

-----

[![arXiv](https://img.shields.io/badge/arXiv-2502.00397-ca001c.svg)](https://arxiv.org/abs/2502.00397) 
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-10888852-99b3eb.svg)](https://ieeexplore.ieee.org/abstract/document/10888852) 

----

![Results Image](./figures/icassp_qual.png)

> Comparing Ground Truth with the predicted saliency maps of our models and STSANet on three different datasets - DHF1K, UCF-Sports and DIEM.

## Abstract

This paper introduces ViNet-S, a 36MB model based on the ViNet architecture with a U-Net design, featuring a lightweight decoder that significantly reduces model size and parameters without compromising performance. Additionally, ViNet-A (148MB) incorporates spatio-temporal action localization (STAL) features, differing from traditional video saliency models that use action classification backbones. Our studies show that an ensemble of ViNet-S and ViNet-A, by averaging predicted saliency maps, achieves state-of-the-art performance on three visual-only and six audio-visual saliency datasets, outperforming transformer-based models in both parameter efficiency and real-time performance, with ViNet-S reaching over 1000fps.

## Architecture

![ViNet Architecture](./figures/arch.png)

## Cite

```bibtex
@INPROCEEDINGS{vinet2025,
  author={Girmaji, Rohit and Jain, Siddharth and Beri, Bhav and Bansal, Sarthak and Gandhi, Vineet},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Minimalistic Video Saliency Prediction via Efficient Decoder & Spatio Temporal Action Cues}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Location awareness;Visualization;Convolution;Streaming media;Predictive models;Transformers;Real-time systems;Decoding;Speech processing;Optimization;Video Saliency Prediction;Efficient Deep Learning;Spatio Temporal Action Cues},
  doi={10.1109/ICASSP49660.2025.10888852},
  url={https://ieeexplore.ieee.org/abstract/document/10888852}
}
```

## Contact

For any queries or questions, please contact [rohit.girmaji@research.iiit.ac.in](mailto:rohit.girmaji@research.iiit.ac.in) or [vgandhi@iiit.ac.in](mailto:vgandhi@iiit.ac.in), or use the public issues section of this repository.

----

> This work © 2025 by the authors of the <a href="https://ieeexplore.ieee.org/abstract/document/10888852">paper</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">