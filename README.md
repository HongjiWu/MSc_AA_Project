# MSc_AA_Project

repo for master project: Assessing the Practical Risks of Text Authorship Attribution Attacks

## Setup

In a new Python 3.7 virtual environment, run:

```bash
pip install -r dev_requirements.txt
```

Create the .env file by copying the .env1 file and changing the data path:
```bash
cp .env1 .env
```

Note: NLTK dependency needed for Narayanan method only.

```python
  >>> import nltk
  >>> nltk.download('averaged_perceptron_tagger')
  >>> nltk.download('universal_tagset')
  >>> nltk.download('stopwords')

  >>> nltk.download('punkt')
```

To use the Stanford Parser (optional and very slow):
```
 cd
 wget http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip
 unzip stanford-parser-full-2015-12-09.zip
 export STANFORDTOOLSDIR=$HOME
 export CLASSPATH=$STANFORDTOOLSDIR/stanford-parser-full-2015-12-09/stanford-parser.jar:$STANFORDTOOLSDIR/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar
```
## Downloading Dataset

For downloading dataset, specify the name of your anchor subreddit, and the minimum training sample each candidate needs to have in **download_dataset.py**

Then run this script, the whole process could take several hours.

## Running experiments

For running Exp.1 Varying Number of Training Samples, run **exp_tr_samples.py**

For running Exp.2 Varying Number of Candidate Authors, run **exp_author_num.py**

For running Exp.3 Varying Length of Data Samples, run **exp_sample_length.py**

Before running this experiment, you have to generate dataset with different sample length by using **modify_sample_length.py**

For running Exp.4 Varying Content Divergence between Referencing and Targetting Data Samples, run **exp_sim.py**

Before running this expereiment, you have to generate dataset with similarity metrics by using **compute_sim.py**
