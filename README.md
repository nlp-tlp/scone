# SConE: Simplified Cone Embeddings with Symbolic Operators for Complex Logical Queries

This is the implementation of the paper
[SConE: Simplified Cone Embeddings with Symbolic Operators for Complex Logical Queries](https://aclanthology.org/2023.findings-acl.755) (Findings of ACL 2023).

## Getting started

### Step 1: Data preparation

- Download the datasets [here](http://snap.stanford.edu/betae/KG_data.zip), then move `KG_data.zip` to `./scone/` directory

- Unzip `KG_data.zip` to `./scone/data/`:

  ```bash
  cd scone/
  unzip -d data KG_data.zip
  ```

### Step 2: Installing requirements

- If you are familiar with `pip|conda`, please install requirements by your own preference:

  ```bash
  python=3.8.11
  pytorch=1.9.1
  numpy=1.19.2
  tqdm=4.65.0
  tensorboardX=2.5.1
  ```

- [Optional] For those prefer to use `Anaconda`, create virtual environment
  named `scone` (default) with dependencies then activate `scone`:

  ```bash
  conda env create -f requirements.yml
  conda activate scone
  ```

### Step 3: Training model

- [Optional] run the following `bash` command to train model for the default dataset `FB15k-237`, uncomment others in `run.sh` to train model using other datasets `(FB15k/NELL995)`:

  ```bash
  ./run.sh
  ```

- Otherwise, use the direct command in the following to train
  `scone` (FB15k-237, etc.).

#### FB15k-237

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
--data_path data/FB15k-237-betae \
--do_train --do_test \
-n 128 -b 512 -d 400 -g 20 \
-lr 0.00005 --max_steps 350001 --cpu_num 2 --geo scone --valid_steps 30000 \
-projm "(1600,2)" --save_checkpoint_steps 30000 -logic "geometry" \
--seed 0 --print_on_screen -p 0.9 -projn "rtrans_mlp" -conj "all" -delta 0.5
```

#### FB15k & NELL995 (see `run.sh`)

## Citation

If you find this code useful for your research, please consider citing the following paper:

```bib
@inproceedings{nguyen2023scone,
    title = "{SC}on{E}: Simplified Cone Embeddings with Symbolic Operators for Complex Logical Queries",
    author = "Nguyen, Chau D. M.  and
      French, Tim  and
      Liu, Wei  and
      Stewart, Michael",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.755",
    pages = "11931--11946",
}
```

## Acknowledgement

We acknowledge the code of [KGReasoning](https://github.com/snap-stanford/KGReasoning) for their contributions.
