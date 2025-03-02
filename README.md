# VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation, CVPR 2025

<a href="https://sakshamsingh1.github.io/vintage/"> 🌐 Project Page </a> | <a href="https://arxiv.org/pdf/2412.10768"> 📖 Paper </a> 

### 🏠 Prepare environment

```bash
conda create -n vintage python=3.9 -y
conda activate vintage
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/sakshamsingh1/vintage_aud_gen.git
bash requirements.sh
```

### 🔈 Inference

#### Download Pretrained checkpoint
```bash
wget -P ckpts https://huggingface.co/datasets/sakshamsingh1/vintage/resolve/main/VTComb_0300000.pt
```

#### Run inference
```bash
# video + generated audio is saved in outputs
bash scripts/sample.sh

# Refer scripts/sample.sh to see more examples.
# CUDA_VISIBLE_DEVICES=0 python sample_VT_comb.py ODE \
# --cross-att-vis-text --s-vis 7.5 --s-txt 7.5 \
# --vid-path assets/airplane_vgg.mp4 \
# --text-prompt "An airplane flies overhead" 
```

### 📚 TODO
- [ ] Training and evaluation.
- [x] Inference code.

## 🤗 Citation
```
@article{kushwaha2024vintage,
  title={Vintage: Joint video and text conditioning for holistic audio generation},
  author={Kushwaha, Saksham Singh and Tian, Yapeng},
  journal={arXiv preprint arXiv:2412.10768},
  year={2024}
}
```

The code is based on [SiT](https://github.com/willisma/SiT), [Tango](https://github.com/declare-lab/tango), [TempoTokens](https://github.com/guyyariv/TempoTokens)