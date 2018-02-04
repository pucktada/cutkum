# Cutkum ['คัดคำ']
Cutkum ('คัดคำ') is a python code for Thai Word-Segmentation using Recurrent Neural Network (RNN) based on Tensorflow library. 

Cutkum is trained on BEST2010, a 5 Millions Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is still a work-in-progress project. Evaluated on the 10% hold-out data from BEST2010 corpus (~600,000 words), the included trained model currently performs at 

98.0% recall, 96.3% precision, 97.1% F-measure (character-level)
93.5% recall, 94.1% precision and 94.0% F-measure (word-level -- same evaluation method as BEST2010)

# Updates
Feb 02, 2018 - add the training script

# Requirements
* python = 2.7, 3.0+
* tensorflow = 1.3

# Installation

`cutkum` can be installed using `pip` and the trained model can be downloaded from github. The current included model (model/lstm.l6.d2.pb) is a stacked bi-directional LSTM neural network with 6 layers. 

```
pip install cutkum

# then download the trained model (either from github) or with wget

wget https://raw.githubusercontent.com/pucktada/cutkum/master/model/lstm.l6.d2.pb
```

# Usages

Once installed, you can use `cutkum` within your python code to tokenize thai sentences. 

```

>>> from cutkum.tokenizer import Cutkum

>>> ck = Cutkum('lstm.l6.d2.pb')
>>> words = ck.tokenize("สารานุกรมไทยสำหรับเยาวชนฯ")

# python 3.0
>>> words
['สารานุกรม', 'ไทย', 'สำหรับ', 'เยาวชน', 'ฯ']

# python 2.7
>>> print("|".join(words)) 
# สารานุกรม|ไทย|สำหรับ|เยาวชน|ฯ

```

You can also use `cutkum` straight from the command line.

```
usage: cutkum [-h] [-v] -m MODEL_FILE
              (-s SENTENCE | -i INPUT_FILE | -id INPUT_DIR)
              [-o OUTPUT_FILE | -od OUTPUT_DIR] [--max | --viterbi]
```

```
cutkum -m model/lstm.l6.d2.pb -s "สารานุกรมไทยสำหรับเยาวชนฯ"

# output as
สารานุกรม|ไทย|สำหรับ|เยาวชน|ฯ
```


`cutkum` can also be used to segment text within a file (with -i), or to segment all the files within a given directory (with -id).

```
cutkum -m model/lstm.l6.d2.pb -i input.txt -o output.txt
cutkum -m model/lstm.l6.d2.pb -id input_dir -od output_dir
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## To Do
* Improve performance, with better better model, and better included trained-model
* Improve the speed when processing big file

