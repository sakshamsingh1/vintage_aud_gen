# VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation, CVPR 2025

<a href="https://sakshamsingh1.github.io/vintage/"> 🌐 Project Page </a> | <a href="https://arxiv.org/pdf/2412.10768"> 📖 Paper </a> | <a href="https://huggingface.co/datasets/sakshamsingh1/vintage/resolve/main/vintage_bench_data.zip"> 🗄️ VinTAGe-Bench Data </a> 

### 🏠 Prepare environment

```bash
conda create -n vintage python=3.10 -y
conda activate vintage

git clone https://github.com/sakshamsingh1/vintage_aud_gen.git
pip install -r requirements.txt
pip install ninja
pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main
```

### 🚀 Gradio demo
You can launch the Gradio interface for VinTAGe by running the following command:
```
python app.py --share
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
- [x] VinTAGe-Bench data.
- [x] Gradio demo.
- [x] Inference code.

## 🤗 Citation
```
@inproceedings{kushwaha2025vintage,
  title={Vintage: Joint video and text conditioning for holistic audio generation},
  author={Kushwaha, Saksham Singh and Tian, Yapeng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={13529--13539},
  year={2025}
}
```

The code is based on [SiT](https://github.com/willisma/SiT), [Tango](https://github.com/declare-lab/tango), [TempoTokens](https://github.com/guyyariv/TempoTokens), [FoleyCrafter](https://github.com/open-mmlab/foleycrafter)