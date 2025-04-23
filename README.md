# Pretraining GPT on the Project Gutenberg Dataset

&nbsp;
## How to Use This Code

&nbsp;

### 1) Download the dataset

In this section, we download books from Project Gutenberg using code from the [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub repository.

As of this writing, this will require approximately 50 GB of disk space and take about 10-15 hours, but it may be more depending on how much Project Gutenberg grew since then.

&nbsp;
#### Download instructions for Linux and macOS users


Linux and macOS users can follow these steps to download the dataset (if you are a Windows user, please see the note below):

1. Set the `03_bonus_pretraining_on_gutenberg` folder as working directory to clone the `gutenberg` repository locally in this folder (this is necessary to run the provided scripts `prepare_dataset.py` and `pretraining_simple.py`). For instance, when being in the `LLMs-from-scratch` repository's folder, navigate into the *03_bonus_pretraining_on_gutenberg* folder via:
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

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

Next, run the `prepare_dataset.py` script, which concatenates the (as of this writing, 60,173) text files into fewer larger files so that they can be more efficiently transferred and accessed:

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **Tip:**
> Note that the produced files are stored in plaintext format and are not pre-tokenized for simplicity. However, you may want to update the codes to store the dataset in a pre-tokenized form to save computation time if you are planning to use the dataset more often or train for multiple epochs. See the *Design Decisions and Improvements* at the bottom of this page for more information.

> **Tip:**
> You can choose smaller file sizes, for example, 50 MB. This will result in more files but might be useful for quicker pretraining runs on a small number of files for testing purposes.


&nbsp;
### 3) Run the pretraining script

You can run the pretraining script as follows. Note that the additional command line arguments are shown with the default values for illustration purposes:

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

The output will be formatted in the following way:

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **Tip:**
> In practice, if you are using macOS or Linux, I recommend using the `tee` command to save the log outputs to a `log.txt` file in addition to printing them on the terminal:

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **Warning:**
> Note that training on 1 of the ~500 Mb text files in the `gutenberg_preprocessed` folder will take approximately 4 hours on a V100 GPU.
> The folder contains 47 files and will take approximately 200 hours (more than 1 week) to complete. You may want to run it on a smaller number of files.
