# Sentiment-Analysis-From-Text

## Introduction

This problem is about how to create an automatic classification from recorded tweets from Twitter. The tweets must be classified by the emotion that is hidden behind the words. For this task three algorithms will be used to find out which one is the most accurate. This coursework is interesting because it combines a lot of the fundamentals of machine learning in one single task which if it was in bigger scale it would be able to save a lot of data processing time and removing hateful content rapidly. 


## Data and Preparation

The first step is importing the csv file “text_emotion_data_filtered.csv” into MATHLAB as a table. In this file we can notice that we have got two columns. The first one is called “sentiment” and contains four types of emotions: enthusiasm, happiness, relief, surprise. The second one is called “Content” and contains the text of the tweet. 
After importing, the data is split into new lines and tokenized. A bag of words is created, and the data is added in it. The “stop words” are removed as well as the words with less than 100 occurrences. A TF-IDF matrix is created using the new bag which is created with the refined data. After that a training and testing vector labels are created as well as training and testing feature matrix. 


## Methodology

To complete this task I had to choose three algorithms out of six. My decision was based on the fact that these algorithms are different in terms of interpretability, memory usage and prediction speed. [1] The chosen algorithm are listed below with references to their functions.
•	The first algorithms is “K-Nearest Neighbour” with its function “fitcknn()”[2] 
•	The second algorithm is “Discriminant Analysis” with its function “fitcdiscr()”[3]
•	The third algorithm is “Naïve Bayes” with its function “fitcnb()”[4]

On each one of them the machine is first trained using the function for each of the algorithms. After that the accuracy is calculated using the formula illustrated below. 
 
Then the matrix with the results is displayed using the function “confusionchart()”.
Results
The charts with the results are displayed on the next page. Figure 1 is for “K-Nearest Neighbour”. Figure 2 is for ”Discriminant Analysis”. And figure 3 is for “Naïve Bayes”. We can notice that the second algorithm was the most accurate with a value of 0.50. or 50% accuracy. The other two are close two each other with values of 0.38 for the first algorithm and 0.30 for the third algorithm.
 
 
## Conclusion

We can conclude that the “Discriminant Analysis algorithm” was the most successful and accurate out of the three tested. For further research would be the other algorithms in the list of the first reference.[1] From our tests we get 50% accuracy as a result. This would not be enough to say that this system is reliable in terms of providing accurate data. There should be further research on this topic implementing different algorithms.


### References

[1] https://uk.mathworks.com/help/stats/supervised-learning-machine-learning-workflow-and-algorithms.html
[2] https://uk.mathworks.com/help/stats/fitcknn.html
[3] https://uk.mathworks.com/help/stats/fitcdiscr.html
[4] https://uk.mathworks.com/help/stats/fitcnb.html



