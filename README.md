# Cutkum
Cutkum (cut-kum = tad-kum or 'ตัดคำ') is a python code for Thai Word-Segmentation using Recurrent Neural Network (RNN) based on Tensorflow library. 

Cutkum is trained on BEST2010 Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is still a work-in-progress project. Evaluated on the 10% hold-out data from BEST2010 corpus, the included trained model currently performs at 0.93 recall, 0.92 precision and 0.93 F-measure. 

# Requirements
* python >= 3.0
* tensorflow 1.1

# Usages
```
usage: cutkum.py [-h] [-v] -c CHECKPOINT_FILE -m META_FILE (-i INPUT_FILE | -s SENTENCE)

```
To run 'Cutkum' takes two model input files, a meta file (-m) and a checkpoint file (-c). 

```
./cutkum.py -m model/ck.r8.s128.l3.meta -c model/ck.r8.s128.l3 -s "สารานุกรมไทยสำหรับเยาวชนฯ เล่มที 12 นี้ ได้พิมพ์ขึ้นครั้งแรกในพ.ศ.2432 มี 10 เรื่อง คือการบูรณะวัดพระศรีรัตนศาสดาราม"
```
which will produce the output word segmentation as followed
```
สารานุกรม|ไทย|สำหรับ|เยาวชน|ฯ| |เล่ม|ที |12| |นี้| |ได้|พิมพ์|ขึ้น|ครั้ง|แรก|ใน|พ.ศ.|2432| |มี| |10| |เรื่อง| |คือ|การ|บูรณะ|วัดพระศรีรัตนศาสดาราม
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

