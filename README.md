# conll2012_boilerplate
Generic code for reading/writing data for the 
<a ref="http://conll.cemantix.org/2012/">CoNLL 2012 Coreference Shared Task</a>

# CoNLL Dataset
The first part of this project is to get the official dataset. Due to copyright it was split in two parts: one can 
be easily downloaded and the other must be downloaded from the LDC site. 
<a href="http://conll.cemantix.org/2012/data.html">More here</a>.
After downloading the files, run the scripts as indicated in the page. The folder structure is a little confusing, but
makes sense for this task.

## Downloading from LDC
The files that are needed are the following:

| LDC Catalog ID | Corpus Name             | File                             |
| -------------- | ----------------------- | -------------------------------- |
| LDC2013T19     | OntoNotes Release 5.0   | ontonotes-release-5.0_LDC2013T19 |
| LDC2012T13     | English Web Treebank    | LDC2012T13.tgz                   |
| LDC2012E74     | CoNLL 2012 Test Set     | LDC2012E74.tgz                   |
| LDC2012E48     | CoNLL 2012 Training Set | LDC2012E48                       |


To download this files you must register on <a href="htts://catalog.ldc.upenn.edu">LDC website</a> and request the 
dataset. If you are linked to a university, search for it. Maybe they already have access. Use this 
<a href="https://catalog.ldc.upenn.edu/LDC2013T19">link</a> to guide you.

#Preparing spacy
The default implementation uses <a href='https://spacy.io/'>spaCy</a> to extract some base information. Run the code 
below once to download what is needed
  
python -m spacy download en

#Running tests
python -m unittest on root folder