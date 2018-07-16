## SERec
Code for AAAI2018 paper "Collaborative Filtering with Social Exposure: A Modular Approach to Social Recommendation"

Contact: Menghan Wang (wangmengh@zju.edu.cn)

## Requirements

- gcc 5.4.0

- gsl 2.2

- openMP (If you don't want to use it, you can delete all the lines look like "#pragma omp parallel for..." in the **"socialexpo.cpp"**)

- (optional) openBLAS

## Data format

A data sample is provided. (the line numbers of files below are used as the index to users or items, starting from 0.)

- user file: total_rated_items [space] Item1:count [space] Item2:count ...

- item file: total_rated_users [space] User1:count [space] User2:count ...

- social file: friend_sum [space] Friend1:1 [space] Friend2:1 ...



## How to use
Note that the SERec is new name of our model, in the code it is named "s_expo".

- Revise the lib path in Makefile to meet your settings. 

- Tune the parameters in run.sh. As there are too many paramters, you may need to tune some parameters in the main.cpp.

- compile the code and run with "run.sh".

- The meaning of the parameter "version":
```
  Verison 1 means exposures are computed purely based on popularity, which is equivalent to "ExpoMF" (Our c++ version is much faster).

  Verison 2 means "social boosting"

  Verison 3 means "social regularization"
```

## Citation
Please cite our paper if it is helpful to your research:
```
@inproceedings{DBLP:conf/aaai/WangZYZ18,
  author    = {Menghan Wang and
               Xiaolin Zheng and
               Yang Yang and
               Kun Zhang},
  title     = {Collaborative Filtering With Social Exposure: {A} Modular Approach
               to Social Recommendation},
  booktitle = {Proceedings of the Thirty-Second {AAAI} Conference on Artificial Intelligence,
               New Orleans, Louisiana, USA, February 2-7, 2018},
  year      = {2018},
  crossref  = {DBLP:conf/aaai/2018},
  url       = {https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16058},
  timestamp = {Thu, 03 May 2018 17:03:19 +0200},
  biburl    = {https://dblp.org/rec/bib/conf/aaai/WangZYZ18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
