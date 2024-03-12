KDD99 dataset is the data set used for The Third International Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99 The Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network intrusion detector, a predictive model capable of distinguishing between ``bad'' connections, called intrusions or attacks, and ``good'' normal connections. This database contains a standard set of data to be audited, which includes a wide variety of intrusions simulated in a military network environment.

The data can be downloaded from http://kdd.ics.uci.edu/databases/kddcup99/

More information can be found here: https://www.ecb.torontomu.ca/~bagheri/papers/cisda.pdf

The data is useful in many ML models for both Supervised Learning, Unsupervised Learning, Anomaly Detection, etc.
The data is "heavy" enough to be considered big data, however, it is processable thru pandas with a little more wait time.

In this project, the KDD99 is used to demonstrate different technique in Statistical Sampling:
(1) Random Sampling
(2) Stratified Random Sampling with Propotionate/Dis-proportionate Sampling technique
(3) Cluster Sampling based on regional/label dataset
