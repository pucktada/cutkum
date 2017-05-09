# Cutkum
Cutkum (cut-kum = tad-kum or 'ตัดคำ') is a software for Thai Word-Segmentation with Recurrent Neural Network (RNN) based on Tensorflow library. Cutkum is trained on BEST2010 Thai words corpus by NECTEC (https://www.nectec.or.th/). It also comes with an already trained model, and can be used right out of the box. Cutkum is currently still a work-in-progress. As for the performance, evaluated on the 10% hold-out data from BEST2010 corpus, the included model currently performs at 0.93 recall, 0.92 precision and 0.93 F-measure. 

# Requirements
--* python 3.0
--* tensorflow 1.1

# Usages

```
./cutkum.py -c model/ck.r8.s128.l3 -m model/ck.r8.s128.l3.meta -s "สารานุกรมไทยสำหรับเยาวชนฯ เล่มที 12 นี้ ได้พิมพ์ขึ้นครั้งแรกในพ.ศ.2432 มี 10 เรื่อง คือการบูรณะวัดพระศรีรัตนศาสดาราม"

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

