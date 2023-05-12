%Importing data as a table
MyTable = readtable('h-drive/361/SentimentAnalysisData/text_emotion_data_filtered.csv');

% Data split on new line and tokenizing the strings in textData
textData = split(MyTable.Content,newline);
docs = tokenizedDocument(textData);

%Creating Bag of Words, adding the document to the bag and removing common words.
bag = bagOfWords;
bag = addDocument(bag,docs);
newBag = removeWords(bag,stopWords);

%Removing any words with fewer than 99 occurrences in the bag
num=99;
newBag1=removeInfrequentWords(newBag,num);


% Tf-idf Matrix calculated for the last created bag.
M1 = tfidf(newBag1);
full(M1)

str=MyTable.sentiment;

%Training Label vector
trLabel1=str(1:6432);

%Training Feature matrix 
tr1M1=M1(1:6432,:);

%Testing Label vector
tsLabel1=str(6432:end);

%Testing Feature matrix
ts1M1=M1(6432:end,:);
%------------------------------------------------------------------
%Here the K-Nearest Neighbour algorithm is used to train the machine
modelknn = fitcknn(tr1M1,trLabel1);
pred = predict(modelknn,ts1M1);

%Accuracy calculations
 correct_predictions_knn=sum(strcmp(pred,tsLabel1));
 accuracy=correct_predictions_knn./numel(tsLabel1);

% Diplay Matrix confusion
 figure(1)
knnmodelchart=confusionchart(tsLabel1,pred);
title(sprintf('Accuracy=(%.2f)',accuracy));
%--------------------------------------------------------
%The Discriminant Analysis algorithm is used to train the machine
modeldiscriminant = fitcdiscr(tr1M1,trLabel1);
pred_discr = predict(modeldiscriminant,ts1M1);

%Calculating accuracy
correct_pred_discr=sum(strcmp(pred_discr,tsLabel1));
accuracy_discr=correct_pred_discr./numel(tsLabel1);

%Display Matrix confusion 
figure(2)
discriminantmodelchart=confusionchart(tsLabel1,pred_discr);
title(sprintf('Accuracy=(%.2f)',accuracy_discr));
%-----------------------------------------------------------
%The Naïve Bayes algorithm is used here to train the machine.
modelnbayes = fitcnb(tr1M1,trLabel1);
 pred_nbayes = predict(modelnbayes,ts1M1);

%Calculating the accuracy
 correct_pred_nbayes=sum(strcmp(pred_nbayes,tsLabel1));
 accuracy_nbayes=correct_pred_nbayes./numel(tsLabel1);

%Display Matrix confusion
 figure(3)
bayesmodelchart=confusionchart(tsLabel1,pred_nbayes);
title(sprintf('Accuracy=(%.2f)',accuracy_nbayes));

