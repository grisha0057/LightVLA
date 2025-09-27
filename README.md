# The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning

**Project website: https://liauto-research.github.io/LightVLA/**

**Paper: https://arxiv.org/abs/2509.12594**

**Example video: https://cloud.tsinghua.edu.cn/f/1e3f4ab2bd7345768a6e/**

This work is built upon the wonderful [OpenVLA-OFT](https://openvla-oft.github.io/) project. Shout out to Moo Jin Kim, Chelsea Finn and Percy Liang.

## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks

Training:
* Between 1-8 GPUs with 27-80 GB, depending on the desired training setup (with default bfloat16 data type). See [this FAQ from OpenVLA-OFT](https://openvla-oft.github.io/#train-compute) for details.

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

## Support

If you run into any issues, please open a new GitHub issue.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2509.12594):

```bibtex
@misc{jiang2025betterlearnsmarterprune,
      title={The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning}, 
      author={Titong Jiang and Xuefeng Jiang and Yuan Ma and Xin Wen and Bailin Li and Kun Zhan and Peng Jia and Yahui Liu and Sheng Sun and Xianpeng Lang},
      year={2025},
      eprint={2509.12594},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.12594}, 
}
```
