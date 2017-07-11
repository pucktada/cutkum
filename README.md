# Cutkum ['คัดคำ']
Cutkum ('คัดคำ') is a python code for Thai Word-Segmentation using Recurrent Neural Network (RNN) based on Tensorflow library. 

Cutkum is trained on BEST2010, a 5 Millions Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is still a work-in-progress project. Evaluated on the 10% hold-out data from BEST2010 corpus (~600,000 words), the included trained model currently performs at 

97.3% recall, 96.7% precision, 97.0% F-measure (character-level)
93.4% recall, 93.9% precision and 93.7% F-measure (word-level -- same evaluation method as BEST2010)

# Requirements
* python >= 2.7
* tensorflow >= 1.1

# Usages
```
usage: cutkum.py [-h] [-v] -m META_FILE -c CHECKPOINT_FILE (-i INPUT_FILE | -s SENTENCE)
```

`cutkum.py` needs two files to load the trained model, a meta_file (the network definition) and a checkpoint_file (the trained weights). `cutkum.py` can be used in two ways, to segment text directly from a given sentence (with -s) or to segment text within a file (with -i)

For example, one can run `cutkum.py` to segment a thai phrase `"สารานุกรมไทยสำหรับเยาวชนฯ"` by running

```
./cutkum.py -m model/ck.r8.s128.l3.meta -c model/ck.r8.s128.l3 -s "สารานุกรมไทยสำหรับเยาวชนฯ"
```

which will produce the resulting word segmentation as followed (words are seperated by '|').

```
สารานุกรม|ไทย|สำหรับ|เยาวชน|ฯ
```

# Citations
Please consider citing this project in your publications if it helps your research.
The following is a [BibTeX](http://www.bibtex.org/) and plaintext reference.
The BibTeX entry requires the `url` LaTeX package.

```
@misc{treeratpituk2017cutkum,
    title        = {{Thai Word-Segmentation with Deep Learning in Tensorflow}},
    author       = {Treeratpituk, Pucktada},
    howpublished = {\url{https://github.com/pucktada/cutkum}},
    note         = {Accessed: [Insert date here]}
}

Pucktada Treeratpituk. Thai Word-Segmentation with Deep Learning in Tensorflow.
https://github.com/pucktada/cutkum.
Accessed: [Insert date here]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## To Do

* Improve performance, with better better model, and better included trained-model
* Providing a script for training a new model (coming soon!, give me some times to refactor the code)
* Providing a script for reproducing the experiment... :)

