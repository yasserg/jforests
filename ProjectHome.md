jforests is a Java library that implements many tree-based learning algorithms.

jforests can be used for regression, classification and ranking problems. The following tutorial shows how jforests can be used for learning a ranking model using the LambdaMART algorithm.

## Learning to Rank with LambdaMART ##

### Data Sets Format ###
jforests uses the following format for its input data sets (same as the one used in SVMLight):

```
<line> .=. <relevance> qid:<qid> <feature>:<value> ... <feature>:<value> 
<relevance> .=. <integer>
<qid> .=. <positive integer>
<feature> .=. <positive integer>
<value> .=. <float>
```

For this tutorial, we will use the sample data set which is available [here](http://jforests.googlecode.com/svn/trunk/jforests/src/main/resources/sample-ranking-data.zip).


### Converting Data Sets to Binary Format ###
In order to speed up the computations, jforests converts its input data sets to binary format. We are assuming that you have unzipped the above sample data set in a folder and are currently on that folder. You should have also [downloaded](http://code.google.com/p/jforests/downloads/list) the latest jforests jar file and renamed it to 'jforests.jar' and put it in the same folder.

The following command can be used for converting data sets to binary format:

```
java -jar jforests.jar --cmd=generate-bin --ranking --folder . --file train.txt --file valid.txt --file test.txt
```

As this command shows, we are converting 'train.txt', 'valid.txt', and 'test.txt' to binary format. As a result 'train.bin', 'valid.bin', and 'test.bin' are generated.

### Learning the Ranking Model ###
Once the input data sets are converted to the binary format, a ranking model can be trained on them.

First you need to specify the parameters of your machine learning algorithm. The following is a sample set of parameters for the LambdaMART algorithm:

```
trees.num-leaves=7
trees.min-instance-percentage-per-leaf=0.25
boosting.learning-rate=0.05
boosting.sub-sampling=0.3
trees.feature-sampling=0.3

boosting.num-trees=2000
learning.algorithm=LambdaMART-RegressionTree
learning.evaluation-metric=NDCG

params.print-intermediate-valid-measurements=true
```

Create a 'ranking.properties' file in the current folder and save the above config in it.

Then the following command can be used for training a LambdaMART ensemble and storing it in the 'ensemble.txt' file:

```
java -jar jforests.jar --cmd=train --ranking --config-file ranking.properties --train-file train.bin --validation-file valid.bin --output-model ensemble.txt
```

### Predicting Scores of Documents ###
Once you have the LambdaMART ensemble, you can use it for predicting scores of test documents. The following command performs this step and stores the results in the 'predcitions.txt' file.

```
java -jar jforests.jar --cmd=predict --ranking --model-file ensemble.txt --tree-type RegressionTree --test-file test.bin --output-file predictions.txt
```

Scores can then be used for measuring NDCG or other information retrieval measures.

### Advanced Ranking Options ###

Jforests can be configured to change the used measure for LambdaMART using the `learning.evaluation-metric` entry in ranking.properties.Currently, NDCG is supported, as well as risk-sensitive evaluation measures such as URisk and TRisk - see [RiskSensitiveLambdaMART](RiskSensitiveLambdaMART.md)

# Source Codes #
Source codes are available for checkout from this subversion repository: [http://jforests.googlecode.com/svn/trunk/](http://jforests.googlecode.com/svn/trunk/)

# Citation Policy #
If you use jforests for a research purpose, please use the following citation:

Y. Ganjisaffar, R. Caruana, C. Lopes, **Bagging Gradient-Boosted Trees for High Precision, Low Variance Ranking Models**, in SIGIR 2011, Beijing, China.

Bibtex:
<pre>
@inproceedings{Ganji:2011:SIGIR,<br>
author = {Yasser Ganjisaffar and Rich Caruana and Cristina Lopes},<br>
title = {Bagging Gradient-Boosted Trees for High Precision, Low Variance Ranking Models},<br>
booktitle = {Proceedings of the 34th international ACM SIGIR conference on Research and development in Information},<br>
series = {SIGIR '11},<br>
year = {2011},<br>
isbn = {978-1-4503-0757-4},<br>
location = {Beijing, China},<br>
pages = {85--94},<br>
numpages = {10},<br>
doi = {http://doi.acm.org/10.1145/2009916.2009932},<br>
acmid = {2009932},<br>
publisher = {ACM},<br>
address = {New York, NY, USA},<br>
}<br>
</pre>