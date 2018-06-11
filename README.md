# Cutkum ['คัดคำ']
Cutkum ('คัดคำ') is a python code for Thai Word-Segmentation using Recurrent Neural Network (RNN) based on Tensorflow library. 

Cutkum is trained on BEST2010, a 5 Millions Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is still a work-in-progress project. Evaluated on the 10% hold-out data from BEST2010 corpus (~600,000 words), the included trained model currently performs at 

98.0% recall, 96.3% precision, 97.1% F-measure (character-level)
RC: 0.988, PC: 0.966, FC: 0.977
95% recall, 95% precision and 95.0% F-measure (word-level -- same evaluation method as BEST2010)

# Update :D

A major update

1. now you dont have to load the model seperately, just do `pip install` and Cutkum is ready to use out of the box.
2. the included model is now smaller, faster, and have higher accuracy. :)

# Requirements
* python = 2.7, 3.0+
* tensorflow = 1.4+

# Installation

`cutkum` can be installed using `pip` 

```
pip install cutkum

```

# Usages

Once installed, you can use `cutkum` within your python code to tokenize thai sentences. 

```

>>> from cutkum.tokenizer import Cutkum

>>> ck = Cutkum()
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
usage: cutkum [-h] [-v]
              (-s SENTENCE | -i INPUT_FILE | -id INPUT_DIR)
              [-o OUTPUT_FILE | -od OUTPUT_DIR] [--max | --viterbi]
```

```
cutkum -s "ล่าสุดกระทรวงพาณิชย์ได้ประกาศตัวเลขการส่งออกของไทย"

# output as
ล่าสุด|กระทรวงพาณิชย์|ได้|ประกาศ|ตัว|เลข|การ|ส่ง|ออก|ของ|ไทย
```


`cutkum` can also be used to segment text within a file (with -i), or to segment all the files within a given directory (with -id).

```
cutkum -i input.txt -o output.txt
cutkum -id input_dir -od output_dir
```

## Citation

```
Pucktada Treeratpituk (2017). Cutkum: Thai Word-Segmentation with LSTM in Tensorflow. May 5, 2017. See https://github.com/pucktada/cutkum
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## To Do
* Improve performance, with better better model, and better included trained-model
* Improve the speed when processing big file

