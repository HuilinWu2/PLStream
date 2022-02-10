# PLStream: A Framework for Fast Polarity Labelling of Massive Data Streams

## Motivation
* When dataset freshness is critical, the annotating of high speed unlabelled data streams becomes critical but remains an open problem.
* We propose PLStream, a novel Apache Flink-based framework for fast polarity labelling of massive data streams, like Twitter tweets or online product reviews.

## Environment Requirements
relative python packages are summerized in `requirements.txt`
1. Flink v1.13
2. Python 3.7
3. Java 8

## DataSource
* Dataset quick access in https://course.fast.ai/datasets#nlp
### Tweets
* 1.6 million labeled Tweets:
* Source:[Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
### Yelp Reviews
* 280,000 training and 19,000 test samples in each polarity
* Source:[Yelp Review Polarity](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)
### Amazon Reviews
* 1,800,000 training and 200,000 testing samples in each polarity
* Source:[Amazon product review polarity](https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz)

## Quick Start
quick try PLStream on yelp review dataset
### Data Prepare
```
cd PLStream
weget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz
tar zxvf yelp_review_polarity_csv.tgz
mv yelp_review_polarity_csv/train.csv train.csv
```
### 1. Install required environment of PLStream
* please make sure Environment Requirements mentioned above is ready.
```
pip install -r requirements.txt
```
### 2. Start Redis-Server in a terminal
```
redis-server
```
### 3. Run PLStream
```
python PLStream.py
```
* The outputs' form is "original text" + "label" + "@@@@":
* With help of a split("@@@@") function we can further reorganize the labelled dataset.

### Optional
to see the labelling accuracy, simply run:
`python PLStream_acc.py`
