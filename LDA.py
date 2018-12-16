#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This portion of the code creates topics and associated words 
using Latent Dirichlet Allocation
@author: halladi@us.ibm.com
"""
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel

sqlContext = SQLContext(sc)
path = "./advisorconversations/advsisortext.txt"

data = sc.textFile(path).zipWithIndex().map(lambda(words,idd): Row(idd= idd, words = words.split(" ")))
docDF = sqlContext.createDataFrame(data)

Vector = CountVectorizer(inputCol="words", outputCol="vectors")
model = Vector.fit(docDF)
result = model.transform(docDF)

corpus_size = result.count()  # total number of words
corpus = result.select("idd", "vectors").map(lambda (x,y): [x,y]).cache()

# Cluster the documents into four topics using LDA
ldaModel = LDA.train(corpus, k=4,maxIterations=100,optimizer='online')
topics = ldaModel.topicsMatrix()
vocabArray = model.vocabulary

wordNumbers = 50  # number of words per topic
topicIndices = sc.parallelize(ldaModel.describeTopics(maxTermsPerTopic = wordNumbers))

def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result
    
topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
  print "Topic" + str(topic)
  for term in topics_final[topic]:
    print term
  print '\n'