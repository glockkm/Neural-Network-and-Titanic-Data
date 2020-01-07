library(nnet)
library(caret)
library(pROC)
library(zoo)

titanic = read.csv(file="tit-train.csv", header = TRUE, sep=",")

summary(titanic)
View(titanic)
str(titanic)

titanic[[7]] #shows the integers and not Null
class(titanic[[7]]) #???? #returns "integer"
class(titanic$SibSP) #????? #returns "NULL"
names(titanic)[7] = "SibSP" #FIXED IT!! Now class(titanic$SibSP) returns "integer"

#preprocessing data
titanic = titanic[ -c(1) ] #take out id column
titanic = titanic[ -c(3) ] #take out name column
titanic = titanic[ -c(7) ] #take out ticket column
titanic = titanic[ -c(8) ] #take out cabin column as more than 60% NA
#titanic = titanic[ -c(5) ] #SibSP column


class(titanic$Pclass)
class(titanic$SibSP) 
class(titanic$Parch)
class(titanic$Fare)
class(titanic$Embarked)

titanic$Survived = as.factor(titanic$Survived)
#0 did not survive and 1 did survive
titanic$Sex = as.numeric(titanic$Sex)
#female is 1 and male is 0
titanic$Embarked = as.numeric(titanic$Embarked)
#2 = C, 3 = Q, 4 = S
titanic$Age = na.aggregate(titanic$Age) #takes mean of column and replaces na with mean. 
#mean of Age column = 29.69912

anyNA(titanic) #true if yes
is.na(titanic) #true if yes a na in each row
titanic[complete.cases(titanic), ] #gives you the rows with no nas in them
sum(is.na(titanic$Age)) #177 total nas #summary( ) also tells us this
177/891 #.20% missing so impute




#split data into train/test how Dr. Cao did in lab 5 example
set.seed(101)
train_split = sample(1:nrow(titanic), nrow(titanic)*0.8) #take 80% sample
train = titanic[train_split, ]
test_without_y = subset(titanic[-train_split, ], select= -Survived)
test_with_y = titanic[-train_split, ]$Survived

#test_x dataset that matches length of train dataset
test = titanic[-train_split, ]
test_x_noClass = test[-1]


#another way to split data using caret package
trainIndex = createDataPartition(titanic$Survived, p=.8, list=F)
train2 = titanic[trainIndex, ] #use in train function for model creation
test2 = titanic[-trainIndex, ] #use in predict function 
test_with_y2 = titanic[-train_split, ]$Survived








#look for best size of hidden layer and decay/learning rate
grid = expand.grid(size=c(2,3,4,5,6,7,8,9,10,11,12,13), decay=c(0, 0.01, 0.1, 1))
model_nn = train(Survived ~., data=train, method="nnet", tunegrid=grid, skip=FALSE, linout=FALSE)
model_nn
#based on accuracy of 0.7975284, the optimal model has a size 3 and decay 0.1

#USE NNET FUNCTION TO BUILD A MODEL USING BEST SIZE AND DECAY
#https://www.rdocumentation.org/packages/nnet/versions/7.3-12/topics/nnet
nnet_model = nnet(Survived ~., data=train, size = 3, decay=0.1, skip=FALSE, linout=FALSE)
nnet_model
pred3 = predict(nnet_model, test_without_y) #nn uses probability to determine the classes
pred3
##https://www.datacamp.com/community/tutorials/confusion-matrix-calculation-r
surv_or_not3 = ifelse(pred3 >= 0.5, "1", "0")
surv_or_not3
#1 is positive survived and 0 is did not survive
#pred is the predict function variable
#convert to factor: fac_class3
fac_class3 = factor(surv_or_not3, levels = levels(test_with_y["Survived"]))
fac_class3
confusionMatrix(fac_class3, test_with_y)





#using caret data split and linout = -0:1
#model_nn2 = train(Survived ~., data=train2, method="nnet", tunegrid=grid, skip=FALSE, linout=FALSE)
#model_nn2
#.7968670 accuracy, size 5, decay 0.1
#pred2 = predict(model_nn2, test2, type="prob") #nn uses probability to determine the classes
#pred2
#confusionMatrix(pred2, test_with_y2)





#run model with normalized data and best size and decay
model_nn_proc = train(Survived ~., data=train, method="nnet", preProcess = c("center", "scale"), tunegrid=grid, skip=FALSE, linout=FALSE)
                                                                               
model_nn_proc
#accuracy = 0.8128359, the optimal size is 1 with a decay of 0.1

#or another way to normalize
#library(BBmisc)
#titanic2 = normalize(titanic, method = "standardize")
#gives -age????? #does not touch factored columns just numeric
#View(titanic2)
#set.seed(101)
#train_split_2 = sample(1:nrow(titanic2), nrow(titanic)*0.8) #take 80% sample
#train_2 = titanic2[train_split_2, ]
#test_x_2 = subset(titanic2[-train_split_2, ], select= -titanic2$Survived)
#test_y_2 = titanic[-train_split, ]$Survived
#model_nn_norm = nnet(Survived ~., data =train_2, size=3, skip=FALSE, linout=FALSE )
#model_nn_norm$decay
#model_nn_norm$nunits
#grid2 = expand.grid(size=c(3), decay=c(0.1))
#model_nn_norm = train(Survived ~., data=train_2, method="nnet", tunegrid=grid2, skip=FALSE, linout=FALSE)
#model_nn_norm
#accuracy = 0.8069231 with size 3 and decay 0.1 
#accuracy slighlty better with normalization (center and scale)


#confusion matrix using test data from split
#pred = predict(model_nn, test_without_y, type="prob") #nn uses probability to determine the classes
#pred
#confusionMatrix(pred, test_with_y)

#Before making the confusion matrix, "cut" the predicted probabilities at a given threshold to turn probabilities into class predictions. 
#do this easily with the ifelse() function
#https://www.datacamp.com/community/tutorials/confusion-matrix-calculation-r
# If pred exceeds threshold of 0.5, M else R: m_or_r
#m_or_r = ifelse(p > 0.5, "M", "R")
#M or 1 positive and r or 0 is negative to match classes in dataset
#surv_or_not = ifelse(pred > 0.5, "1", "0")
#surv_or_not
#1 is positive survived and 0 is did not survive
#pred is the predict function variable

# Convert to factor: fac_class
#fac_class = factor(surv_or_not, levels = levels(test_y["Survived"]))
#fac_class

# Create confusion matrix
#confusionMatrix(fac_class, test_y["Survived"])
#confusionMatrix(fac_class, test_y)
###arguments not same length?????
###is test_x longer than test_y? In total instances yes!
#fac_class has 358 instances
#test_y has 179
#test_x has 179 #put in pred function
#train has 712 instances
###maybe split data differently???

#another way to convert probabilities to 0 and 1??
#pred$`0`[pred$`0`>=.50] = 0  
#pred$`0`[pred$`0`<.50] = ? 
#pred$`0`
#pred$`1`[pred$`1`>=.50] = 1  
#pred$`1`
#levels are not overlapping (probability in pred versus 0 and 1 in data.
#need to factor the same?????
#class(titanic$Survived)
#class(pred$`0`)
#sum(pred$`0`)
#sum(pred$`1`)
#test_y

#pred$`0` = (pred$`1` = 0)
#pred$`1` = (pred$`1` = 1)
#pred$`0`
#pred$`1`


#using tit-test dataset test the nn model and calculate how many survived and how many did not
titanic3 = read.csv(file = "tit-test.csv", header=TRUE, sep=",")
View(titanic3)
titanic3 = titanic3[ -c(1) ] #take out id column
titanic3 = titanic3[ -c(2) ] #take out name column
titanic3 = titanic3[ -c(6) ] #take out ticket column
titanic3 = titanic3[ -c(7) ] #take out cabin column as more than 60% NA
titanic3 = titanic3[ -c(4) ] #take out SibSP column until I can figure out "NULL

titanic3$Sex = as.numeric(titanic3$Sex)
#female is 1 and male is 0
titanic3$Embarked = as.numeric(titanic3$Embarked)
#2 = C, 3 = Q, 4 = S
titanic3$Age = na.aggregate(titanic3$Age) #takes mean of column and replaces na with mean. 

pred_test = predict(model_nn, titanic3, type="prob")
pred_test


survived = pred_test[which(pred_test[,2]>.50),]
#164 survived
did_not_survive = pred_test[which(pred_test[,1]>.50),]
#253 did not survive
