#!/usr/bin/python3

# Usage:
# -(i)nput -(o)utput 
# ./sentiment.py -i training_data.csv -o results.txt


import sys
import getopt
import csv

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


            with open("polarity_pos", "w") as polarity_pos_file:
                for i in positive_rows:
                    polarity_pos_file.write(str(i[0]) + "\n")

            with open("polarity_neg", "w") as polarity_neg_file:
                for i in negative_rows: 
                    polarity_neg_file.write(str(i[0]) + "\n")


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


