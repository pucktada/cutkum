# Cutkum ['คัดคำ']
Cutkum ('คัดคำ') is a python code for Thai Word-Segmentation using Recurrent Neural Network (RNN) based on Tensorflow library. 

Cutkum is trained on BEST2010, a 5 Millions Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is still a work-in-progress project. Evaluated on the 10% hold-out data from BEST2010 corpus (~600,000 words), the included trained model currently performs at 

98.0% recall, 96.3% precision, 97.1% F-measure (character-level)

93.5% recall, 94.1% precision and 94.0% F-measure (word-level -- same evaluation method as BEST2010)

# Requirements
* python = 2.7, 3.0+
* tensorflow >= 1.1

# Usages
```
usage: cutkum.py [-h] [-v] -m MODEL_FILE
                 (-d DIRECTORY | -i INPUT_FILE | -s SENTENCE) [-o OUTPUT_DIR]
                 [--max | --viterbi]

```

`cutkum.py` needs to load the trained model, the current included model (model/lstm.l6.d2.pb) is a bi-directional LSTM neural network with 6 layers. `cutkum.py` can be used in 3 ways, 1. to segment text directly from a given sentence (with -s), 2. to segment text within a file (with -i), and 3. to segment all the files within a given directory.

For example, one can run `cutkum.py` to segment a thai phrase `"สารานุกรมไทยสำหรับเยาวชนฯ"` by running

```
./cutkum.py -m model/lstm.l6.d2.pb -s "สารานุกรมไทยสำหรับเยาวชนฯ"
```

which will produce the resulting word segmentation as followed (words are seperated by '|').

```
สารานุกรม|ไทย|สำหรับ|เยาวชน|ฯ
```

or if one want to segment a text file 'input.txt' and save the result to 'output.txt'

```
./cutkum.py -m model/lstm.l6.d2.pb -i input.txt > output.txt
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## To Do

* Improve performance, with better better model, and better included trained-model
* Improve the speed when processing big file

