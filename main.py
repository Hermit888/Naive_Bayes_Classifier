import pandas as pd
import os, math, sys
from nltk.corpus import stopwords
from nltk.tokenize import SpaceTokenizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics import confusion_matrix

# enter python main.py "TRAIN_SIZE" in terminal to start
# for example "python main.py 80"


#### get train data size and test data size
try:
    train_size = int(sys.argv[1])
except: 
    train_size = 80

if train_size < 20 or train_size > 80 or train_size == None or not str(train_size).isdigit():
    train_size = 80

print(
    'Qian, Yuxuan, A20484572 solution: \
    \nTraining set size: ' + str(train_size) + '%\
    \n\nData clean-up...')



#### data clean-up (remove words is non-english and emotion symbols)
df = pd.read_csv('reviews.csv', usecols=['content', 'score'])

## find contents with non-English 
idx_lst = []
idx = 0
stopwds = stopwords.words('english')
for i in df['content']:
    # check if the sentence with non-english
    if not i.isascii():
        words = i.split()
        newsen = []
        # find word that is non-english and remove it
        for j in words:
            if j.isascii():
                newsen.append(j)

        # replace to new sentence
        df.replace(i, ' '.join(newsen), inplace=True)



print('Text pre-processing...')
##### text-preprocessing (remove punctuation)
# function for removing punctuation
def removePunc(content):
    tk = WordPunctTokenizer()
    tokens = tk.tokenize(content)
    newsent = []

    for w in tokens:
        # check if w in stop words corpus
        if w not in stopwds and (w.isalpha() or w.isdigit()):
            newsent.append(w)
    
    return " ".join(newsent).lower()


idx = 0
for i in df['content']:
    df.replace(i, removePunc(i), inplace=True) # lowercase


# remove empty contents and reset idx for df
for i in range(len(df)):
    if df.at[i, 'content'] == '':
        df.drop(i, inplace=True)

df = df.reset_index(drop=True)




print('Training classifier...')
#### get train and test data set
train = df.head(round(len(df)*(train_size/100)))
test = df.tail(round(len(df)*0.2)).reset_index(drop=True)



#### compute probabilities for each category
p1 = len(train.groupby('score').get_group(1)) / len(train)
p2 = len(train.groupby('score').get_group(2)) / len(train)
p3 = len(train.groupby('score').get_group(3)) / len(train)
p4 = len(train.groupby('score').get_group(4)) / len(train)
p5 = len(train.groupby('score').get_group(5)) / len(train)
prob_list = [p1, p2, p3, p4, p5]



#### extract vocabulary from train data (get vocabulary V)
V = set()
tk = SpaceTokenizer()

for i in train['content']:
    V = V.union(set(tk.tokenize(i)))



#### probaility for each word in that class
# dictionary for each labeling classes
label1, label2, label3, label4, label5 = {str(key): 0 for key in V}, {str(key): 0 for key in V}, {str(key): 0 for key in V}, {str(key): 0 for key in V}, {str(key): 0 for key in V}

# total counts of words in each class
totalWords = {str(key):0 for key in range(1, 6)}

## get words that exist in that labeling category
# binary bag of words 
for idx in range(len(train)):
    if train.at[idx, 'score'] == 1:
        # get total words in each class
        totalWords['1'] += len(set(tk.tokenize(train.at[idx, 'content']))) 
        
        # get each word count for that category
        for w in set(tk.tokenize(train.at[idx, 'content'])): # binary so only check if word exist or not
            # for category rating score 1, the word exist so +1
            label1[w] += 1
    
    elif train.at[idx, 'score'] == 2:
        # get total words in that class
        totalWords['2'] += len(set(tk.tokenize(train.at[idx, 'content']))) 
        
        # get each word count for that category
        for w in set(tk.tokenize(train.at[idx, 'content'])):
            label2[w] += 1    
    
    elif train.at[idx, 'score'] == 3:
        # get total words in that class
        totalWords['3'] += len(set(tk.tokenize(train.at[idx, 'content']))) 
        
        # get each word count for that category
        for w in set(tk.tokenize(train.at[idx, 'content'])):
            label3[w] += 1    

    elif train.at[idx, 'score'] == 4:
        # get total words in that class
        totalWords['4'] += len(set(tk.tokenize(train.at[idx, 'content']))) 
        
        # get each word count for that category
        for w in set(tk.tokenize(train.at[idx, 'content'])):
            label4[w] += 1        

    else:
        # get total words for that class
        totalWords['5'] += len(set(tk.tokenize(train.at[idx, 'content'])))
        
        # get each word count for that category
        for w in set(tk.tokenize(train.at[idx, 'content'])):
            label5[w] += 1


## adding 1 smoothing and computing log probability for each categories
for k, v in label1.items(): 
    label1[k] = (v + 1) / (totalWords['1'] + 1)

for k, v in label2.items():
    label2[k] = (v + 1) / (totalWords['2'] + 1)

for k, v in label3.items():
    label3[k] = (v + 1) / (totalWords['3'] + 1)

for k, v in label4.items():
    label4[k] = (v + 1) / (totalWords['4'] + 1)

for k, v in label5.items():
    label5[k] = (v + 1) / (totalWords['5'] + 1)



print('Testing classifier...\n')
#### test classifier
labels = [label1, label2, label3, label4, label5]

# function for determining content's label
# log probability to classify
def labelIs(initprob, lst, text):
    score_list = []
    
    for i in range(len(lst)):
        curr_prob = math.log(initprob[i])
        for w in text:
            if w in lst[i]:
                curr_prob += math.log(lst[i][w])
        
        score_list.append(curr_prob)
    
    return score_list.index(max(score_list)) + 1

# predict the label
prediction = []
for i in range(len(test)):
    prediction.append(labelIs(prob_list, labels, set(tk.tokenize(test.at[i, 'content'].lower()))))



### Test classifier
# get confusion matrix and amount in each cell
confuMatrix = confusion_matrix(test['score'].tolist(), prediction, labels=[1, 2, 3, 4, 5])
s11, s12, s13, s14, s15 = confuMatrix[0][0], confuMatrix[0][1], confuMatrix[0][2], confuMatrix[0][3], confuMatrix[0][4]
s21, s22, s23, s24, s25 = confuMatrix[1][0], confuMatrix[1][1], confuMatrix[1][2], confuMatrix[1][3], confuMatrix[1][4]
s31, s32, s33, s34, s35 = confuMatrix[2][0], confuMatrix[2][1], confuMatrix[2][2], confuMatrix[2][3], confuMatrix[2][4]
s41, s42, s43, s44, s45 = confuMatrix[3][0], confuMatrix[3][1], confuMatrix[3][2], confuMatrix[3][3], confuMatrix[3][4]
s51, s52, s53, s54, s55 = confuMatrix[4][0], confuMatrix[4][1], confuMatrix[4][2], confuMatrix[4][3], confuMatrix[4][4]
# print(confuMatrix)

print("Test results / metrics:")
## TP, FN, FP, TN for each class
# score 1
TP, FN = s11, (s12 + s13 + s14 + s15)
FP = (s21 + s31 + s41 + s51)
TN = s22 + s23 + s24 + s25 + \
     s32 + s33 + s34 + s35 + \
     s42 + s43 + s44 + s45 + \
     s52 + s53 + s54 + s55

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
negPred = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('\nNumber of true positive for labeling score 1: ' + str(TP) + '\
    \nNumber of true negatives for labeling score 1: ' + str(TN) + '\
    \nNumber of false positives for labeling score 1: ' + str(FP) + '\
    \nNumber of false negatives for labeling score 1: ' + str(FN) + '\
    \nSensitivity (recall) for labeling score 1: ' + str(sensitivity) + '\
    \nSpecificity for labeling score 1: ' + str(specificity) + '\
    \nPrecision for labeling score 1: ' + str(precision) + '\
    \nNegative predictive value for labeling score 1: ' + str(negPred) + '\
    \nAccuracy for labeling score 1: ' + str(accuracy) + '\
    \nF-score for labeling score 1: ' + str(2 * precision * sensitivity / (precision + sensitivity)) + '\n')


#score 2
TP, FN = s22, (s21 + s23 + s24 + s25)
FP = (s12 + s32 + s42 + s52)
TN = s11 + s13 + s14 + s15 + \
     s31 + s33 + s34 + s35 + \
     s41 + s43 + s44 + s45 + \
     s51 + s53 + s54 + s55

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
negPred = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('Number of true positive for labeling score 2: ' + str(TP) + '\
    \nNumber of true negatives for labeling score 2: ' + str(TN) + '\
    \nNumber of false positives for labeling score 2: ' + str(FP) + '\
    \nNumber of false negatives for labeling score 2: ' + str(FN) + '\
    \nSensitivity (recall) for labeling score 2: ' + str(sensitivity) + '\
    \nSpecificity for labeling score 2: ' + str(specificity) + '\
    \nPrecision for labeling score 2: ' + str(precision) + '\
    \nNegative predictive value for labeling score 2: ' + str(negPred) + '\
    \nAccuracy for labeling score 2: ' + str(accuracy) + '\
    \nF-score for labeling score 2: ' + str(2 * precision * sensitivity / (precision + sensitivity)) + '\n')


# score 3
TP, FN = s33, (s31 + s32 + s34 + s35)
FP = (s13 + s23 + s43 + s53)
TN = s11 + s12 + s14 + s15 + \
     s21 + s22 + s24 + s25 + \
     s41 + s42 + s44 + s45 + \
     s51 + s52 + s54 + s55

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
negPred = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('Number of true positive for labeling score 3: ' + str(TP) + '\
    \nNumber of true negatives for labeling score 3: ' + str(TN) + '\
    \nNumber of false positives for labeling score 3: ' + str(FP) + '\
    \nNumber of false negatives for labeling score 3: ' + str(FN) + '\
    \nSensitivity (recall) for labeling score 3: ' + str(sensitivity) + '\
    \nSpecificity for labeling score 3: ' + str(specificity) + '\
    \nPrecision for labeling score 3: ' + str(precision) + '\
    \nNegative predictive value for labeling score 3: ' + str(negPred) + '\
    \nAccuracy for labeling score 3: ' + str(accuracy) + '\
    \nF-score for labeling score 3: ' + str(2 * precision * sensitivity / (precision + sensitivity)) + '\n')


#score 4
TP, FN = s44, (s41 + s42 + s43 + s45)
FP = (s14 + s24 + s34 + s54)
TN = s11 + s12 + s13 + s15 + \
     s21 + s22 + s23 + s25 + \
     s31 + s32 + s33 + s35 + \
     s51 + s52 + s53 + s55

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
negPred = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('Number of true positive for labeling score 4: ' + str(TP) + '\
    \nNumber of true negatives for labeling score 4: ' + str(TN) + '\
    \nNumber of false positives for labeling score 4: ' + str(FP) + '\
    \nNumber of false negatives for labeling score 4: ' + str(FN) + '\
    \nSensitivity (recall) for labeling score 4: ' + str(sensitivity) + '\
    \nSpecificity for labeling score 4: ' + str(specificity) + '\
    \nPrecision for labeling score 4: ' + str(precision) + '\
    \nNegative predictive value for labeling score 4: ' + str(negPred) + '\
    \nAccuracy for labeling score 4: ' + str(accuracy) + '\
    \nF-score for labeling score 4: ' + str(2 * precision * sensitivity / (precision + sensitivity)) + '\n')


# score 5
TP, FN = s55, (s51 + s52 + s53 + s54)
FP = (s15 + s25 + s35 + s45)
TN = s11 + s12 + s13 + s14 + \
     s21 + s22 + s23 + s24 + \
     s31 + s32 + s33 + s34 + \
     s41 + s42 + s43 + s44

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
negPred = TN / (TN + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print('\nNumber of true positive for labeling score 5: ' + str(TP) + '\
    \nNumber of true negatives for labeling score 5: ' + str(TN) + '\
    \nNumber of false positives for labeling score 5: ' + str(FP) + '\
    \nNumber of false negatives for labeling score 5: ' + str(FN) + '\
    \nSensitivity (recall) for labeling score 5: ' + str(sensitivity) + '\
    \nSpecificity for labeling score 5: ' + str(specificity) + '\
    \nPrecision for labeling score 5: ' + str(precision) + '\
    \nNegative predictive value for labeling score 5: ' + str(negPred) + '\
    \nAccuracy for labeling score 5: ' + str(accuracy) + '\
    \nF-score for labeling score 5: ' + str(2 * precision * sensitivity / (precision + sensitivity)) + '\n')



### Ask the user for keyboard input 
# function for calculating two labels' prob
def calculateProb(initprob, lst, text):
    score_list = []
    
    for i in range(len(lst)):
        curr_prob = initprob[i]
        for w in text:
            if w in lst[i]:
                curr_prob *= lst[i][w]
        
        score_list.append(curr_prob)
    
    return score_list[0], score_list[1], score_list[2], score_list[3], score_list[4]

# enter sentence
cont = 'Y'
tk = SpaceTokenizer()
while cont == 'Y': 
    S = input('Enter your sentence:')

    print('\nSentence S: \
        \n' + S)

    print('was classified as score ' + str(labelIs(prob_list, labels, set(tk.tokenize(removePunc(S))))) + '.')

    prob1, prob2, prob3, prob4, prob5 = calculateProb(prob_list, labels, set(tk.tokenize(removePunc(S))))

    print("P(score 1 | S) = " + str(prob1))
    print("P(score 2 | S) = " + str(prob2))
    print("P(score 3 | S) = " + str(prob3))
    print("P(score 4 | S) = " + str(prob4))
    print("P(score 5 | S) = " + str(prob5) + '\n')

    cont = input("Do you want to enter another sentence [Y/N]? ")
    print()