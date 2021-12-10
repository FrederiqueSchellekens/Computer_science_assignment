# Computer_science_assignment
This assignment concerns LSHMSM.py

This code implements a Local Sensitivity Hashing algorithm and Model-Component Similarity Method on previous research.
With the objective of finding duplicates within a data set consisting of tv's from several web shops, whilst reducing the computational burden which would typically arise from this problem. This leads to the objective of this code, testing the scalability of the implemented methods on large data sets.

The code exists of one file. Within this code, the seperate definition for the preprocessing of the large data sets into a training and test data set. And transforming these data sets seperately into signatures of low dimensionality which still contain a lot of information about the products. The Local Sensitivity Hashing definition then uses the signatures found for all products to construct the potential duplicates. The MSM definition then compares these potential duplicates on several similarity measures and finally clusters the potential duplicates based on the dissimilarity between the pairs. Finally the output definition constructs several measures to evaluate the methods on. These include among others the F_1 score. 
A general part is written beneath all definitions where the definitions are called. Here a bootstrap loop is ran 5 times for the data set. No hyperparameters are optimized within the model. 
Output for measures is finally plotted.

To use this code, the data set in the first lines of the code can be adjusted to contain the data set as prefered. However this data set should be a json file, consisting of products. For which the modelId is known and they all contain a title.
'data 1' is an adjusted json file to allow for looping over the dictionary by printing '{"TVs": [ ' before the json file. and ]} after the json file.
