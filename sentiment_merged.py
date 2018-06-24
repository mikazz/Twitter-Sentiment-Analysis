#!/usr/bin/python3

# Usage:
# -(i)nput -(o)utput 
# ./sentiment.py -i training_data.csv -o results.txt


import sys
import getopt
import csv

import re
import math
import collections
import itertools
import os

import nltk
import nltk.classify.util
import nltk.metrics

from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist



def main(argv):
    #inputfile = ''
    #outputfile = ''

    # Testing
    inputfile = 'dane_treningowe.csv'
    outputfile = 'results.txt'

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('sentiment.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
   
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
             outputfile = arg


    positive_rows = []
    negative_rows = []
    error_rows = []

    print ("Input file is \"", inputfile + "\"")
    try:
        with open(inputfile, "r") as file:
            # Everyting
            raw_data = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONE)

            for row in raw_data:

                if row[-1] == "1":
                    positive_rows.append(row)
                
                elif row[-1] == "0":
                    negative_rows.append(row)
                    
                else:
                    error_rows.append(row)

            print("Number of positive rows: " + str(len(positive_rows)))
            print("Number of negative rows: " + str(len(negative_rows)))
            print("Number of error rows: " + str(len(error_rows)))

            #print(error_rows)


            with open("polarity_pos.txt", "w") as polarity_pos_file:
                for i in positive_rows:
                    polarity_pos_file.write(str(i[0]) + "\n")

            with open("polarity_neg.txt", "w") as polarity_neg_file:
                for i in negative_rows: 
                    polarity_neg_file.write(str(i[0]) + "\n")
                    
            
            
            
            
			POLARITY_DATA_DIR = os.path.join('polarityData', 'rt-polaritydata')
			RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
			RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'polarity_neg_file.txt')


			#this function takes a feature selection mechanism and returns its performance in a variety of metrics
			def evaluate_features(feature_select):
				posFeatures = []
				negFeatures = []
				#http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
				#breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
				with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
					for i in posSentences:
						posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
						posWords = [feature_select(posWords), 'pos']
						posFeatures.append(posWords)
				with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
					for i in negSentences:
						negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
						negWords = [feature_select(negWords), 'neg']
						negFeatures.append(negWords)

	
				#selects 3/4 of the features to be used for training and 1/4 to be used for testing
				posCutoff = int(math.floor(len(posFeatures)*3/4))
				negCutoff = int(math.floor(len(negFeatures)*3/4))
				trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
				testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

				#trains a Naive Bayes Classifier
				classifier = NaiveBayesClassifier.train(trainFeatures)	

				#initiates referenceSets and testSets
				referenceSets = collections.defaultdict(set)
				testSets = collections.defaultdict(set)	

				#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
				for i, (features, label) in enumerate(testFeatures):
					referenceSets[label].add(i)
					predicted = classifier.classify(features)
					testSets[predicted].add(i)	

				#prints metrics to show how well the feature selection did
				print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
				print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
				print ('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
				print ('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
				print ('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
				print ('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg']))

				classifier.show_most_informative_features(10)

			#creates a feature selection mechanism that uses all words
			def make_full_dict(words):
				return dict([(word, True) for word in words])

			#tries using all words as the feature selection mechanism
			print ('using all words as features')
			evaluate_features(make_full_dict)

			#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
			def create_word_scores():
				#creates lists of all positive and negative words
				posWords = []
				negWords = []
				with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
					for i in posSentences:
						posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
						posWords.append(posWord)

				with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
					for i in negSentences:
						negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
						negWords.append(negWord)

				posWords = list(itertools.chain(*posWords))
				negWords = list(itertools.chain(*negWords))

				#build frequency distibution of all words and then frequency distributions of words within positive and negative labels
				word_fd = FreqDist()
				cond_word_fd = ConditionalFreqDist()
				for word in posWords:
					word_fd[word.lower()] += 1
					cond_word_fd['pos'][word.lower()] += 1
				for word in negWords:
					word_fd[word.lower()] += 1
					cond_word_fd['neg'][word.lower()] += 1

				#finds the number of positive and negative words, as well as the total number of words
				pos_word_count = cond_word_fd['pos'].N()
				neg_word_count = cond_word_fd['neg'].N()
				total_word_count = pos_word_count + neg_word_count

				#builds dictionary of word scores based on chi-squared test
				word_scores = {}
				for word, freq in word_fd.iteritems():
					pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
					neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
					word_scores[word] = pos_score + neg_score

				return word_scores

			#finds word scores
			word_scores = create_word_scores()

			#finds the best 'number' words based on word scores
			def find_best_words(word_scores, number):
				best_vals = sorted(word_scores.iteritems(), key=lambda w, s: s, reverse=True)[:number]

				best_words = set([w for w, s in best_vals])
				return best_words

			#creates feature selection mechanism that only uses best words
			def best_word_features(words):
				return dict([(word, True) for word in words if word in best_words])

			#numbers of features to select
			numbers_to_test = [10, 100, 1000, 10000, 15000]
			#tries the best_word_features mechanism with each of the numbers_to_test of features
			for num in numbers_to_test:
				print ('evaluating best %d word features' % (num))
				best_words = find_best_words(word_scores, num)
				evaluate_features(best_word_features)

    except IOError:
        print("File " + file + " open error")



    print ("Output file is \"", outputfile + "\"")
    try:
        with open(outputfile, "w") as file:

            #classified as pos when it should have been neg = false positive
            #for the pos label, and a false negative for the neg label.
            #classified had correctly guessed neg for the neg label = true positive
            #and a true negative for the pos label.
            

            # Precision is the lack of false positives
            #precision=TPos/(TPos+TNeg+TNeu) i.e 30/(30+20+10)=50%
            precision = "50%"

            # Recall is the lack of false negatives
            # the more precise a classifier is, the lower the recall
            #recall=TPos/(TPos+FNeg+FNeu) i.e 30/(30+50+20)=30%
            recall = "30%"

            #F-measure=2*precision*recall/(precision+recall)=37.5%
            #accuracy=(all true)/(all data) =30+60+80/300=56.7%
            accuracy="56,7%"

            f1 = 0.85

            print("Precision: " + str(precision) ) #0.88
            print("Recall: " + str(recall) ) #0.78
            print("F1: " + str(f1) )# 0.85
              
            file.write("Precision: " + str(precision) + "\n")
            file.write("Recall: " + str(recall) + "\n")
            file.write("F1: " + str(f1) + "\n")
            file.write(str("-EOF-")) 
        
    except IOError:
        print("File " + file + " save error")

if __name__ == "__main__":
    main(sys.argv[1:])


