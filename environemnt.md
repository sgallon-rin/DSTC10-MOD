```
conda create -n dstc10 python=3.8 && conda activate dstc10
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.21.3 numpy==1.22.4 tqdm sentencepiece scikit-learn
```

```
export TOKENIZERS_PARALLELISM=false
```
