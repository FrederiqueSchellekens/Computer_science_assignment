import json
import numpy as np
import random
import itertools as itertools
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# import the data
with open(r'C:\Users\frank\Documents\AMM\TVs-all-merged.json') as Jsonfile:
    data = json.load(Jsonfile)
with open(r'C:\Users\frank\Documents\AMM\TVs-all-merged1.json') as Jsonfile:
    data1 = json.load(Jsonfile)

# create a list with all the keys of the TVs-all-merged
temp_list = []
for key in data:
    temp_list.append(key)

titles_1 = []
titles = ''
keys = []
# create list of all titles, create a list of all the modelID's and create a string of all the titles after another
for j in range(len(temp_list)):
    for item in data1['TVs']:
        for name in item[temp_list[j]]:
            title = name['title']
            key = name['modelID']
            titles_1.append(title)
            titles = (titles + ' ' + title)
            keys.append(key)

X = titles.split()  # split the string titles in seperate words
MW = list(dict.fromkeys(X))  # remove duplicates from the list of words to create the vector of model words


# create a function 'preprocessing', in this fuction we perform bootstrap, create binary representation vectors,
# and hash the binary vectors to a signature matrix.
def preprocessing(titles_1, keys):
    # divide the data in a training and test set
    def bootstrap(titles_1, keys):
        length = len(titles_1)
        total = [*range(0, length, 1)]

        training = random.choices(total, weights=None, cum_weights=None, k=length)
        training = list(dict.fromkeys(training))  # remove all duplicates
        training = list(dict.fromkeys(training))

        test = list(set(total) - set(training))  # list with all numbers not in training

        # CREATE JSON FOR BOTH DATASETS
        testset1 = []
        testset2 = []
        trainingset1 = []
        trainingset2 = []
        for i in range(0, length):
            if i in test:
                testset1.append(keys[i])
                testset2.append(titles_1[i])
            else:
                trainingset1.append(keys[i])
                trainingset2.append(titles_1[i])

        return testset1, testset2, trainingset1, trainingset2

    test_keys, test_title_1, training_keys, training_title_1 = bootstrap(titles_1, keys)

    # create the matrix of binary vectors
    def binaryvectors(keys, titles_1):
        titles = ''
        for i in range(0, len(titles_1)):
            titles = (titles + ' ' + titles_1[i])  # create a string of all the titles after another

        X = titles.split()
        MW = list(dict.fromkeys(X))  # create the vector of model words

        # create binary columns
        k = 0
        b = np.zeros((len(MW), len(titles_1)))
        for i in titles_1:
            Y = i.split()
            for j in range(len(MW)):
                for x in Y:
                    if MW[j] == x:
                        b[j, k] = 1
            k = k + 1
        return b, MW

    test_b = binaryvectors(test_keys, test_title_1)
    training_b = binaryvectors(training_keys, training_title_1)

    # minhashing
    def createhash(b, MW, titles_1):
        # create permutations
        random.seed(49)
        p1 = random.sample(range(len(MW)), len(MW))
        p2 = random.sample(range(len(MW)), len(MW))
        p3 = random.sample(range(len(MW)), len(MW))
        p4 = random.sample(range(len(MW)), len(MW))
        p5 = random.sample(range(len(MW)), len(MW))
        p6 = random.sample(range(len(MW)), len(MW))
        p7 = random.sample(range(len(MW)), len(MW))
        p8 = random.sample(range(len(MW)), len(MW))
        p = [p1, p2, p3, p4, p5, p6, p7, p8]

        # PERFORM THE MINHASHING
        # this funtion creates the signature column of one product
        def create_hash(vector: list):
            signature = []
            for j in range(0, 8):
                for i in range(1, len(MW)):
                    idx = p[j].index(i)
                    signature_val = vector[idx]
                    if signature_val == 1:
                        signature.append(i)
                        break
            return signature

        # loop over all products and create the complete signature matrix.
        signature2 = np.zeros((8, len(titles_1)))
        for x in range(0, len(titles_1)):
            signature2[:, x] = create_hash(b[:, x])
        return (temp_list, keys, titles_1, signature2)

    # create the training and test signature matrix
    trainingsignature = createhash(training_b[0], training_b[1], training_title_1)
    testsignature = createhash(test_b[0], test_b[1], test_title_1)

    return trainingsignature, testsignature, test_keys, test_title_1, training_keys, training_title_1


trainingsignature1, testsignature2, test_keys, test_title_1, training_keys, training_title_1 = preprocessing(titles_1,
                                                                                                             keys)


# this function performs the LSH
def LSH_complete(signature2, r):
    # split the signature in multiple bands
    def splitsignature(signature, r):
        b = int(len(signature) / r)
        subvector = []
        for i in range(0, len(signature), r):
            subvector.append(signature[i: i + r])
        return subvector

    # perform the actual LSH
    def LSH(signature, r):
        buckets = {}
        length = len(signature[1])
        # for every signature/product
        for j in range(0, length, 1):
            bandj = splitsignature(signature[:, j], r)  # create a band
            # hash products to bands
            for i in range(0, len(bandj)):
                band = bandj[i]
                bandedj = ""
                for t in range(0, len(band)):
                    X = band[t]
                    bandedj = bandedj + str(X)[:-2]
                if bandedj in buckets.keys():
                    buckets[bandedj].append(j)
                else:
                    buckets.setdefault(bandedj, [])
                    integer_to_append = j
                    buckets[bandedj].append(integer_to_append)
        key_list_buckets = []

        for key in buckets:
            key_list_buckets.append(key)

        # create the candidate pairs
        z = 0
        candidate_pairs = []
        for i in range(0, len(key_list_buckets)):
            length = len(buckets[key_list_buckets[i]])
            if length <= 1:
                next
            else:
                Matches = list(itertools.combinations(buckets[key_list_buckets[i]], 2))
                for j in range(0, len(Matches)):
                    Match = str(Matches[j])
                    candidate_pairs.append(Match)
                    z += 1

        candidate_pairs = list(dict.fromkeys(candidate_pairs))
        return candidate_pairs, key_list_buckets

    # create the candidate pairs and the keys of the buckets
    buckets = LSH(signature2, r)
    return buckets


# this function generates the final pairs,the dictionary of clusters and the keys of this dictionary
def MSM(theta1, theta2, candidate_pairs, key_list_buckets, data1, temp_list, signature2):
    titles_1 = []
    titles = ''
    keys = []

    # create list of all titles and keys
    for j in range(len(temp_list)):
        for item in data1['TVs']:
            for key in item:
                keys.append(key)
            for name in item[temp_list[j]]:
                title = name['title']
                titles_1.append(title)
                titles = (titles + title)

    # this function returns the number of duplicates in a list
    def duplicates_in_list(x, y):
        count = 0
        for num in y:
            if num in x:
                count += 1
        return count

    # calculate how similar two strings are based on qgrams and the dice coefficient
    def qgrams(a, b):
        elements_a = []
        elements_b = []
        for i in range(0, len(a) - 2):
            A = a.replace(' ', '')  # delete spaces of the string
            B = b.replace(' ', '')
            elements_a.append(A[i:i + 3])
            elements_b.append(B[i:i + 3])
        dup = duplicates_in_list(elements_a, elements_b)  # calculate number of duplicates between the two lists
        lgth = len(elements_a) + len(elements_b)
        dice_coef = 2 * dup / lgth  # calculate dice coeff
        return dice_coef

    # create the dissimilarity matrix
    def dissimilarity_matrix_func(theta1, theta2, candidate_pairs, key_list_buckets, data1, temp_list, signature2):
        dissimilarity_matrix = np.ones((len(signature2[1]), len(signature2[1]))) * 100000
        # calcuate the similarity of each pair
        for t in range(0, len(candidate_pairs)):  # loop over elk pair
            # obtaining the indices of the potential pairs
            candidatepair = candidate_pairs[t]
            candidatepair = candidatepair.split(", ")
            candidatepair_1 = candidatepair[0].split("(")
            candidatepair_2 = candidatepair[1].split(")")
            candidatepair1 = int(candidatepair_1[1])
            candidatepair2 = int(candidatepair_2[0])

            # set value of the similarity for each pair to zero
            hsim = 0
            # calculate how many similar model words in title
            for item in data1['TVs']:
                for name in item[temp_list[candidatepair1]]:
                    X = name['title']
                    X.split()  # get individual words of the titles
                for name in item[temp_list[candidatepair2]]:
                    Y = name['title']
                    Y.split()
                num_dup = duplicates_in_list(X, Y)
                perc_matching_mw = num_dup / (len(X) + len(Y))  # calculatie the percentage of same model words in title
                hsim = theta1 * perc_matching_mw  # add to the hsim with weight theta1
            # calculate the similarity of the titles without spaces with qgrams
            for item in data1['TVs']:
                for name in item[keys[candidatepair1]]:
                    X = name['title']
                    X.replace(' ', '')
                for name in item[keys[candidatepair2]]:
                    Y = name['title']
                    Y.replace(' ', '')
                SIM2 = qgrams(X, Y)  # calculate the similarity with qgrams
                hsim = hsim + theta2 * SIM2  # add to the hsim with weight theta2
            idx = candidatepair1
            idy = candidatepair2
            dissimilarity_matrix[idx, idy] = 1 - hsim  # add the dissimilarity to the dissimilarity matrix
        return dissimilarity_matrix

    # create the output for the dissimilarity matrix
    dissimilarity_matrix = dissimilarity_matrix_func(theta1, theta2, candidate_pairs, key_list_buckets, data1,
                                                     temp_list, signature2)

    # cluster the dissimilarity matrix
    epsilon = 0.5
    clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
                                       distance_threshold=epsilon)
    clustered = clusters.fit(dissimilarity_matrix)
    cluster = clustered.fit_predict(dissimilarity_matrix)

    # create a dictionary with the clusters
    dictionary = {}
    for i in range(len(cluster)):
        if cluster[i] in dictionary:
            dictionary[cluster[i]].append(i)
        else:
            dictionary.setdefault(cluster[i], [])
            dictionary[cluster[i]].append(i)

    # calculate the final pairs
    finalpairs = {}
    z = 0
    test_keys = []
    for i in dictionary:
        length = len(dictionary[i])
        if length <= 1:
            next
        else:
            Matches = list(itertools.combinations(dictionary[i], 2))
            test_keys.append(key)
            for j in range(0, len(Matches)):
                finalpairs[z] = Matches[j]
                z += 1
    return dictionary, finalpairs, test_keys


# create output of the model
def output(keys, found_pairs):
    # find the true duplicates of the dataset
    def findtrueduplicates(keys):
        trueMatches = {}
        finalduplicates = {}
        for i in range(0, len(keys)):
            if keys[i] in trueMatches.keys():
                trueMatches[keys[i]].append(i)
            else:
                trueMatches.setdefault(keys[i], [])
                trueMatches[keys[i]].append(i)
        z = 0
        for i in trueMatches:
            length = len(trueMatches[i])
            if length <= 1:
                next
            else:
                possiblematches = int((length * (length - 1)) / 2)
                Matches = list(itertools.combinations(trueMatches[i], 2))
                for j in range(0, possiblematches):
                    finalduplicates[z] = Matches[j]
                    z += 1
        return finalduplicates

    # calculate the F1 score, true positives, false positives, false negatives
    def F1_score(found_pairs, true_duplicates):
        truepositive = 0
        falsepositive = 0
        found_pairs2 = list(found_pairs)
        for i in range(0, len(found_pairs2)):
            k = 0
            found_pair = found_pairs[i]
            true_duplicates2 = list(true_duplicates)
            for j in range(0, len(true_duplicates2)):
                true_pair = true_duplicates[j]
                duplicated = found_pair
                trueduplicate = true_pair
                if duplicated == trueduplicate:
                    k += 1
            if k >= 1:
                truepositive += 1
            else:
                falsepositive += 1
        falsenegative = len(found_pairs) - truepositive - falsepositive
        F1_score = truepositive / (truepositive + 0.5 * (falsepositive + falsenegative))
        return (F1_score, truepositive)

    # create true duplicates, F1, true positives, false positives, false negatives
    true_duplicates = findtrueduplicates(keys)
    F_1 = F1_score(found_pairs, true_duplicates)

    return F_1[0], F_1[1], true_duplicates


# in the next part the whole proces is performed 5 times. This gives the average evaluation measurements and data to provide the plots.
# initialize parameters
s=0
repeats = 5
F_1_total=[]
paircompleteness=[]
F_1_star=[]
fractioncomparisons=[]
pairquality=[]
r_options= np.arange(2, 10, 2)
for j in range(0,4):
    r=r_options[s]
    F1_scoreavg = []
    lentrue = []
    n=[]
    amountfound_total = []
    amountcompared = []
    for i in range(0, repeats):
        # create test and training sets and signature matrices
        trainingsignature1, testsignature2, test_keys, test_title_1, training_keys, training_title_1 = preprocessing(titles_1, keys)
        temp_list_train=trainingsignature1[0]
        temp_list_test=testsignature2[0]

        # perform LSH on train and test set
        LSH_train = LSH_complete(trainingsignature1[3], r)
        LSH_test = LSH_complete(testsignature2[3], r)

        # generate potential duplicates for train and test
        potential_duplicates_train = LSH_train[0]
        potential_duplicates_test = LSH_test[0]
        
        key_list_buckets_train = LSH_train[1]
        key_list_buckets_test = LSH_test[1]

        # perform MSM on train and test set
        theta1 = 0.5
        theta2 = 0.5
     
        MSM_train = MSM(theta1, theta2, potential_duplicates_train, key_list_buckets_train, data1, temp_list_train, trainingsignature1[3])
        MSM_test = MSM(theta1, theta2, potential_duplicates_test, key_list_buckets_test, data1, temp_list_test, testsignature2[3])
        finalpairs_train = MSM_train[1]
        finalpairs_test=MSM_test[1]

        # create output
        F_1, amountfound, amounttrue = output(test_keys, finalpairs_test)
    
        F1_scoreavg.append(F_1)
        lentrue.append(len(amounttrue))
        amountfound_total.append(amountfound)
        amountcompared.append(len(potential_duplicates_test))
        n.append(len(test_title_1))

    # create final output
    F1 = np.mean(F1_scoreavg)
    F_1_total.append(F1)
    avg_n=np.mean(n)
    avg_comparisons=np.mean(amountcompared)
    fractioncomparisons.append(avg_comparisons/(avg_n*(avg_n-1)/2))
    
    avg_found= np.mean(amountfound_total)
    pairquality_1=avg_found/avg_comparisons
    pairquality.append(pairquality_1)

    avg_true=np.mean(lentrue)
    paircompleteness_1=avg_found/avg_true
    paircompleteness.append(paircompleteness_1)
    
    star_mean=(2*pairquality_1*paircompleteness_1)/(pairquality_1+paircompleteness_1)
    F_1_star.append(star_mean)
    
    s=s+1

print("frac", fractioncomparisons)
print("PQ", pairquality)
print("PC", paircompleteness)
print("F_1",  F_1_total)
print("F_1*", F_1_star)

# make the plots
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()


ax1.plot(fractioncomparisons, pairquality)
ax1.set_xlabel('fractioncomparisons')
ax1.set_ylabel('pairquality')


ax2.plot(fractioncomparisons, paircompleteness)
ax2.set_xlabel('fractioncomparisons')
ax2.set_ylabel('paircompleteness')

ax3.plot(fractioncomparisons, F_1_total)
ax3.set_xlabel('fractioncomparisons')
ax3.set_ylabel('F_1')

ax4.plot(fractioncomparisons, F_1_star)
ax4.set_xlabel('fractioncomparisons')
ax4.set_ylabel('F_1*')

plt.show()
