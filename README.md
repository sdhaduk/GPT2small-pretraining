# Pretraining GPT on the Project Gutenberg Dataset

&nbsp;
## How to Use This Code

&nbsp;

### 1) Download the dataset

Download books from Project Gutenberg using code from the [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub repository.

This will require approximately 60 GB of disk space and take about 10-15 hours, but it may be more depending on how much Project Gutenberg grew since this writing.

&nbsp;
#### Download instructions for Linux and macOS users

1. Set the `GPT2small-pretraining` folder as working directory to clone the `gutenberg` repository locally in this folder (this is necessary to run the provided scripts `prepare_dataset.py` and `pretraining_simple.py`).


2. Clone the `gutenberg` repository in there:
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. Navigate into the locally cloned `gutenberg` repository's folder:
```bash
cd gutenberg
```

4. Install the required packages defined in *requirements.txt* from the `gutenberg` repository's folder:
```bash
pip install -r requirements.txt
```

5. Download the data:
```bash
python get_data.py
```

6. Go back into the `03_bonus_pretraining_on_gutenberg` folder
```bash
cd ..
```


### 2) Prepare the dataset

Next, run the `prepare_dataset.py` script, which concatenates text files into fewer larger files so that they can be more efficiently transferred and accessed. You dont have to add these arguments when running the script as they are default values, and you can change them to fit your folder structure.

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_tokenized
```


&nbsp;
### 3) Run the pretraining script

You can run the pretraining script as follows. Note that the additional command line arguments are shown with the default values for illustration purposes. You dont have to add these arguments when running the script as they are default values, and you can change them to fit your folder structure. If you do change the batch size make sure you also change the learning rates to account for that.

```bash
python pretrain.py \
  --data_dir "gutenberg_tokenized" \
  --n_epochs 1 \
  --batch_size 8 \
  --output_dir model_checkpoints
```

You can interrupt training and resume it at exactly where you left off my providing the --resume_from arg like so: 
```bash
python pretrain.py \
--resume_from "model_checkpoints/checkpoint_stepxyz.pt" \
```

You can choose to train on a subset of the tokenized files if you wish by modifying the all_files variable under "_ name _ =  _ main _"
