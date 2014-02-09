# Random forest for each data set.

library(randomForest)
set.seed(1279)

num = commandArgs(TRUE)[1]

for(i in num:num) {
    trainFile <- paste("../data/train/ACT", i, "_competition_training.csv", sep='')
    #testFile <- paste("../data/test/ACT", i, "_competition_test.csv", sep='')
  
    # using colClasses to speed up reading of files
    train <- read.csv(trainFile, header=TRUE, nrows=100)
    classes = sapply(train, class)
    train <- read.csv(trainFile, header=TRUE, colClasses=classes)
  
    result <- rfcv(train[, 3:length(train)], train$Act, cv.fold=5, scale="step", step=-200)
    print(result$error.cv)

    #rf <- randomForest(train[, 3:length(train)], train$Act, ntree=100, do.trace=2, mtry=25)
    #test <- read.csv(testFile, header=TRUE)
    #result <- predict(rf, test[, 2:length(test)], type="response")

    #submission <- data.frame(test$MOLECULE, result)
    #colnames(submission) <- c("MOLECULE", "Prediction")
}

#write.csv(submission, paste("../submission/rf_", num, ".csv", sep=''), row.names=FALSE)
