# Random forest for each data set.

library(glmnet)
set.seed(1010)

num = commandArgs(TRUE)[1]

for(i in num:num) {
    trainFile <- paste("../data/train/ACT", i, "_competition_training.csv", sep='')
    testFile <- paste("../data/test/ACT", i, "_competition_test.csv", sep='')
  
    # using colClasses to speed up reading of files
    train <- read.csv(trainFile, header=TRUE, nrows=100)
    classes <- sapply(train, class)
    train <- read.csv(trainFile, header=TRUE, colClasses=classes)

    cvglm <- cv.glmnet(as.matrix(train[, 3:length(train)]), as.matrix(train$Act), nfolds=10, type.measure="mse")
    pdf(file=paste("plots/", "glmnet_", num, ".pdf", sep=''))
    plot(cvglm, ylim=c(0, 1))
    dev.off()
    bestlambda <- cvglm$lambda.min

    # Build model
    model <- glmnet(as.matrix(train[, 3:length(train)]), as.matrix(train$Act), family="gaussian", lambda=bestlambda)

    # Predict
    test <- read.csv(testFile, header=TRUE)
    result <- predict(model, newx=as.matrix(test[, 2:length(test)]))

    submission <- data.frame(test$MOLECULE, result)
    colnames(submission) <- c("MOLECULE", "Prediction")
    write.csv(submission, paste("../submission/glmnet_", num, ".csv", sep=''), row.names=FALSE)
}

