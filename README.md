# anticipation-on-reading-times
Code accompanying paper "On the Effect of Anticipation on Reading Times".

## Install Dependencies

First, create a conda environment with
```bash
$ conda env create -f scripts/environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```

Also, install the necessary nltk sub-package. In a Python sheel, run:
```python
$ import nltk
$ nltk.download('punkt')
```

Finally, install the required R libraries with:
```bash
$ Rscript scripts/r_installer.R
```

## Running project's analysis

First, get the Dundee data `dundee.zip` and add it to folder `corpora`.
Then, get the data for the other datasets in this project by running:
```bash
$ make get_data
```

After that, you can easily run the entire pipeline for a dataset by running:
```bash
$ make DATASET=<dataset>
```
where dataset can be one of: `brown`, `natural_stories`, `dundee`, `provo`, `dundee_skip2zero`, `provo_skip2zero` (the last two of which include skipped words).

For plotting the results in hte paper, then run:
```bash
$ make plot_results
```
The plots should be available in folder `results/plots/`.

Finally, each of the paper's tables can be displayed by running:
```bash
$ make print_table_x
```
where `x` is an integer from 1 to 8.


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bash
@article{pimentel-etal-2022-effect,
    title = "On the Effect of Anticipation on Reading Times",
    author = "Pimentel, Tiago and
    Meister, Clara and
    Wilcox, Ethan G. and
    Levy, Roger and
    Cotterell, Ryan",
    journal = "arXiv preprint",
    year = "2022",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/anticipation-on-reading-times/issues).