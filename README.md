# HmmBiasResolver

## Introduction
HmmBiasResolver is an Hmm-based model for resolving the bias in Electronic Medical Records (EMR), therefore, improving the performance of EMR data analytics.

## Functionality
HmmBiasResolver takes raw array-like data as input, fills the missing data with an Hmm-based model and then outputs the transformed data in the same shape.
### Input:
* raw array-like data with shape (n_patients, n_timewindows, n_features)
* value range {-1, 0, +1}, with -1 "abnormal", "0" missing, "+1" normal
### Output:
* transformed array-like data with shape (n_patients, n_timewindows, n_features)
* value range [-1, +1]

## Requirements

```
hmmlearn==0.2.0
numpy==1.13.3
progressbar==2.3
```

## Reference
K. Zheng, J. Gao, K. Y. Ngiam, B. C. Ooi, and W.L.J. Yip.  
**Resolving the Bias in Electronic Medical Records.**  
*Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (SIGKDD), pages 2171-2180, 2017.* 
