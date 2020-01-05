#########################################################################################
######### R code in supplement to submission on multilevel prediction models#############
######### Authors: Shuai Yuan, Brigitte Kroon & Astrid Kramer ###########################
######### Part 2 code (execution of predcition methods) will be... ######################
######### avilable upon acceptance of the original article ##############################
#########################################################################################

#########################################################################################
# designated packages should be uploaded before analyses
#########################################################################################
library(dplyr)
library("naniar")
library("tibble")
library("glmmLasso")
library("lme4")
library(lavaan)
library(Hmisc)
library(Metrics)
library(caret)
library("randomForest")
library("rpart")
library("car")
library(Matrix)
library(lattice)
library(nlme)
library(doParallel)
library("reticulate")
library("FSA")
library(MASS)
library(REEMtree)
library("matrixcalc")
library(groupdata2)
library(party)
library(apaTables)

#########################################################################################
# part1: pre-processing on original datasets (only for review purpose)
#########################################################################################
##data read-in
sample2 <- read.csv(file = "data complete.csv", header = TRUE)
#dataset cleaning and summarizing
#individual level
sample2 <- sample2[,-1]
sample2$lmx <- apply(sample2[,27:38], 1, mean)
sample2$infshare <- apply(sample2[,39:40], 1, mean)
sample2$voice <- apply(sample2[,41:44], 1, mean)
sample2$orgid <- apply(sample2[,54:59], 1, mean)
sample2$caropp <- apply(sample2[,60:62], 1, mean)
sample2$paysatis <- apply(sample2[,63:65], 1, mean)
sample2$fair <- apply(sample2[,66:81], 1, mean)
sample2$proact <- apply(sample2[,82:86], 1, mean)
#########################################
# group level
sample2$TI <- apply(sample2[,17:19], 1, mean)
sample2$ce <- apply(sample2[,97:103], 1, mean)   
sample2$sp <- apply(sample2[,104:116], 1, mean)
sample2$eo <- apply(sample2[,117:125], 1, mean)

####################################################################
## select the variables in two levels
variables.level2 <- sample2 %>%
  dplyr::select(N, FTEN, FTELY, Council, HR, Levels, Departments,  Managers, FB,  
                HR1:HR5, HR6:HR9, SB, AB, ce, sp, eo)

variables.level1 <- sample2 %>%
  dplyr::select(TI, code, age, gender, edu,
                manage, conhour, contype, lmx, infshare, voice, paysatis, fair, proact, caropp)

both.level <- cbind(variables.level1, variables.level2)

## centering and standardizing the variables
variables.level1[,3:ncol(variables.level1)] <- apply(variables.level1[,3:ncol(variables.level1)], 2, function(x) scale(x, scale = FALSE))
both.level[,3:ncol(both.level)] <- apply(both.level[,3:ncol(both.level)], 2, function(x) scale(x, scale = FALSE))

## save the inter-media dataset
complete.data <- list(variables.level1,  both.level)
save(complete.data, file = "complete.RData")

####################################################################
## the following code can be used to create data partitions
## total number of repetitions
rep <- 200
## lists to store data partitions
train_set.f.s <- list()
train_set.b.s <- list()
test_set.f.s <- list()
test_set.b.s <- list()
folds.s <- list()
train_set.f.r <- list()
train_set.b.r <- list()
test_set.f.r <- list()
test_set.b.r <- list()
folds.r <- list()
## create the data partitions
set.seed(970912)
for (i in 1:rep){
  ####################################################################
  ## create the partitions in such a way that all observations from the validation set belong to existing groups
  ## create the validation set and the training set
  all_data <- groupdata2::partition(variables.level1, p = .09, id_col = "code", list_out = FALSE)
  partition.i <- all_data$.partitions
  nrow.test <- sum(partition.i == 1)
  nrow.train <- sum(partition.i == 2)  
  train_set <- all_data %>%
    filter(.partitions == 2)
  test_set <- all_data %>%
    filter(.partitions == 1)  
  ## from the training set, create the 10 folds used in 10-fold cross validation
  ##create the partitions for the datasets including only individual-level predictors
  random_train_set <- fold(train_set, k = 10, id_col = "code")
  folds.i <-random_train_set$.folds
  random_train_set$TI <- as.vector(scale(random_train_set$TI, scale = FALSE))
  test_set$TI <- as.vector(scale(test_set$TI, scale = FALSE))
  test_set <- within(test_set, rm(.partitions))
  random_train_set <- within(random_train_set, rm(.partitions))
  test_set.f.s[[i]] <- test_set
  train_set.f.s[[i]] <- random_train_set
  
  ##create the partitions for the datasets including predictors from both levels 
  train_set <- both.level[partition.i == 2,]
  test_set <- both.level[partition.i == 1,]
  train_set$TI <- as.vector(scale(train_set$TI, scale = FALSE))
  test_set$TI <- as.vector(scale(test_set$TI, scale = FALSE))
  train_set$.folds <- folds.i
  test_set.b.s[[i]] <- test_set
  train_set.b.s[[i]] <- train_set 
  
############################################################

  ## create the partitions in such a way that all observations from the validation set not belong to existing groups
  ## use iterative procedures to guarantee that no group ids appear in both the training set and the validation set 
  conv <- 1
  index.i <- sample(nrow(variables.level1), size = nrow.test)
  while(conv){
    index.code <- variables.level1[index.i, ]$code
    rest.code <- variables.level1[-index.i, ]$code
    rest.index <- setdiff(1:nrow(variables.level1), index.i)
    new.data <- index.i[!index.code %in% rest.code]
    if(length(new.data) == 0){
      conv <- 0
    }
    if(length(new.data) != 0){
      shift.index <- sample(rest.index, size = length(new.data))
      index.int <- setdiff(index.i, new.data)
      index.i <- c(index.int, shift.index)
    }    
  }
  test_set <- variables.level1[index.i,]
  train_set <- variables.level1[-index.i,]
  random_train_set <- fold(train_set, k = 10, id_col = "code")
  folds.i <-random_train_set$.folds
  random_train_set$TI <- as.vector(scale(random_train_set$TI, scale = FALSE))
  test_set$TI <- as.vector(scale(test_set$TI, scale = FALSE))
  test_set.f.r[[i]] <- test_set
  train_set.f.r[[i]] <- random_train_set
  
  train_set <- both.level[-index.i,]
  test_set <- both.level[index.i,]
  train_set$TI <- as.vector(scale(train_set$TI, scale = FALSE))
  test_set$TI <- as.vector(scale(test_set$TI, scale = FALSE))
  train_set$.folds <- folds.i
  test_set.b.r[[i]] <- test_set
  train_set.b.r[[i]] <- train_set 
}

# create the list to store the indices of fold assignment
n.folds <- 10
for (i in 1:rep){
  data.set <- train_set.b.r[[i]]
  folds.r[[i]] <- list()
  for(j in 1:n.folds){
    folds.r[[i]][[j]] <- which(data.set$.folds != j)
  }
  data.set <- train_set.b.s[[i]]
  folds.s[[i]] <- list()
  for(j in 1:n.folds){
    folds.s[[i]][[j]] <- which(data.set$.folds != j)
  }
}

## data storage 
partition.data <- list(train_set.b.r = train_set.b.r, test_set.b.r = test_set.b.r, 
                       train_set.b.s = train_set.b.s, test_set.b.s = test_set.b.s,  
                       train_set.f.r = train_set.f.r, test_set.f.r = test_set.f.r, 
                       train_set.f.s = train_set.f.s, test_set.f.s = test_set.f.s, 
                       fold.r = folds.r, fold.s = folds.s)

save(partition.data, file = "partition random.RData")

#########################################################################################
# part2: model estimation (data processing is only for review purpose)
#########################################################################################
rep <- 200
n.datasets <- 4
n.methods <- 8
## list to store the results (the prediction performance as well as the variable importance matrices)
rmse.results.list <- list()
for (i in 1:n.datasets){
  rmse.results.list[[i]] <-  matrix(0, nrow = rep, ncol = 6 * n.methods)
}
#########variable importance matrices for each model
for (j in 1:n.methods){
  assign(paste0("var.importance.model", j, sep = ""), lapply(1:n.datasets, function(i) i))
  for (m in 1:n.datasets){
    if(m == 1){
      test_set <- test_set.f.r
    }
    if(m == 2){
      test_set <- test_set.b.r
    }
    if(m == 3){
      test_set <- test_set.f.s
    }
    if(m == 4){
      test_set <- test_set.b.s
    }
    eval(parse(text=c(paste0("var.importance.model", j, "[[m]]<-matrix(nrow = 200, ncol = ncol(test_set[[1]]) - 2)", sep = ""))))
    eval(parse(text = paste0("colnames(var.importance.model", j, "[[m]]) <- colnames(test_set[[1]])[-c(1,2)]", sep = "")))
  }
}

################################################################################################
## read in the data
load("partition random.RData")
train_set.b.s <- partition.data$train_set.b.s
test_set.b.s <- partition.data$test_set.b.s
train_set.f.s <- partition.data$train_set.f.s
test_set.f.s <- partition.data$test_set.f.s
train_set.b.r <- partition.data$train_set.b.r
test_set.b.r <- partition.data$test_set.b.r
train_set.f.r <- partition.data$train_set.f.r
test_set.f.r <- partition.data$test_set.f.r
folds.r <- partition.data$fold.r
folds.s <- partition.data$fold.s 
#################################################################################################

##################################################################################################
# Here starts model estimations for eight models
# the two unpenalized linear models (model 1-2) as well as the two (single) tree models (model 5-6)...
# are trained without parallel computation, while the other four models (model 3-4; 7-8) are trained with... 
# parallel computation
#########################################################################################
## model: linear regression
for (m in 1:n.datasets){
  rmse.results <- rmse.results.list[[m]]
  var.importance <- var.importance.model1[[m]]
  if(m == 1){
    test_set <- test_set.f.r
    train_set <- train_set.f.r    
  }
  if(m == 2){
    test_set <- test_set.b.r
    train_set <- train_set.b.r  
  }
  if(m == 3){
    test_set <- test_set.f.s
    train_set <- train_set.f.s 
  }
  if(m == 4){
    test_set <- test_set.b.s
    train_set <- train_set.b.s
  }
  for (i in 1:rep){
    train_set.i <- train_set[[i]]
    train_set.i <- within(train_set.i, rm(code, .folds))
    test_set.i <- test_set[[i]]
    test_set.i <- within(test_set.i ,rm(code))
    # training of model 1
    model1a <- lm(TI ~ ., data = train_set.i)
    # obtain the relative importance metrices
    imp <- abs(summary(model1a)$coefficients[,3])
    for (p in 2:ncol(train_set.i)){
      if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
        var.importance[i,(p-1)]  <- imp[which(names(imp) == colnames(train_set.i)[p])]
      }
    }
    predict.1a <- predict(model1a, test_set.i, allow.new.levels = TRUE)
    rmse.results[i, 1] <- Metrics::rmse(predict.1a, test_set.i$TI)
    rmse.results[i, 2] <- Metrics::mae(predict.1a, test_set.i$TI)
    rmse.results[i, 3] <- rcorr(predict.1a, test_set.i$TI)$r[2,1] ^ 2
    predict.1a.w <- predict(model1a, train_set.i, allow.new.levels = TRUE)
    rmse.results[i, 25] <- Metrics::rmse(predict.1a.w, train_set.i$TI)
    rmse.results[i, 26] <- Metrics::mae(predict.1a.w, train_set.i$TI)
    rmse.results[i, 27] <- rcorr(predict.1a.w, train_set.i$TI)$r[2,1] ^ 2
  }
 rmse.results.list[[m]] <- rmse.results
 var.importance.model1[[m]] <- var.importance
}
#########################################################################
## mmodel2: random effects linear model
for (m in 1:n.datasets){
  rmse.results <- rmse.results.list[[m]]
  var.importance <- var.importance.model2[[m]]
  if(m == 1){
    test_set <- test_set.f.r
    train_set <- train_set.f.r    
  }
  if(m == 2){
    test_set <- test_set.b.r
    train_set <- train_set.b.r  
  }
  if(m == 3){
    test_set <- test_set.f.s
    train_set <- train_set.f.s 
  }
  if(m == 4){
    test_set <- test_set.b.s
    train_set <- train_set.b.s
  }
  for (i in 1:rep){
    train_set.i <- train_set[[i]]
    train_set.i <- within(train_set.i, rm(.folds))
    test_set.i <- test_set[[i]]
    n <- names(train_set.i)
    f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
    # training of model 2
    model2 <- lmer(f,  data = train_set.i)
    # obtain the relative importance metrices
    imp <- abs(summary(model2)$coefficients[,3])
    for (p in 3:ncol(train_set.i)){
      if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
        var.importance[i,(p-2)]  <- imp[which(names(imp) == colnames(train_set.i)[p])]
      }
    }
    predict.2 <- predict(model2, newdata = test_set.i, allow.new.levels = TRUE)
    rmse.results[i, 4] <- Metrics::rmse(predict.2, test_set.i$TI)
    rmse.results[i, 5] <- Metrics::mae(predict.2, test_set.i$TI)
    rmse.results[i, 6] <- rcorr(predict.2, test_set.i$TI)$r[2,1] ^ 2
    predict.2.w <- predict(model2, newdata = train_set.i, allow.new.levels = TRUE)
    rmse.results[i, 28] <- Metrics::rmse(predict.2.w, train_set.i$TI)
    rmse.results[i, 29] <- Metrics::mae(predict.2.w, train_set.i$TI)
    rmse.results[i, 30] <- rcorr(predict.2.w, train_set.i$TI)$r[2,1] ^ 2
  }
  rmse.results.list[[m]] <- rmse.results
  var.importance.model2[[m]] <- var.importance
}

###########################################################################
## model5: (single) tree models
set.seed(970912)
rep <- 200
## optional specification of the working directory to store the results of the estimation
setwd("~/tree")
for (m in 1:n.datasets){
  rmse.results <- rmse.results.list[[m]]
  var.importance <- var.importance.model3[[m]]
  if(m == 1){
    test_set <- test_set.f.r
    train_set <- train_set.f.r    
  }
  if(m == 2){
    test_set <- test_set.b.r
    train_set <- train_set.b.r  
  }
  if(m == 3){
    test_set <- test_set.f.s
    train_set <- train_set.f.s 
  }
  if(m == 4){
    test_set <- test_set.b.s
    train_set <- train_set.b.s
  }
  for (i in 1:rep){
    train_set.i <- train_set[[i]]
    train_set.i <- within(train_set.i, rm(code, .folds))
    test_set.i <- test_set[[i]]
    test_set.i <- within(test_set.i ,rm(code))
    if(m %in% 1:2){
      part.index <- folds.r[[i]]
    }
    if(m %in% 3:4){
      part.index <- folds.s[[i]]
    }
    
    # training of model 5 with specified assignments of folds 
    # the trainControl function paired with the train function is a universal solution to train any machine learning algorithm that is included in caret
    my_control <- trainControl(method = "cv", 
                               number = 10, 
                               # the index indicating training folds is specified
                               index = part.index,
                               allowParallel = FALSE)
    
    model3 <- caret::train(TI ~ ., data = train_set.i, 
                           method = "rpart", 
                           trControl = my_control,
                           ## cp is the tuning parameter in tree models.
                           ## the grid of cp is kept constant in two (single) tree models
                           tuneGrid = expand.grid(cp = seq(.001, .031, .002)),
                           metric = "RMSE",
                           # the control argument here is used to control some less important paramters in training the trees
                           # (the most important parameter is "cp", which is optimized in the training process)
                           control = rpart.control(minsplit = 10, maxdepth = 30))
    
    # obtain the relative importance metrices
    imp <- varImp(model3)$importance
    for (p in 2:ncol(train_set.i)){
      if (length(which(row.names(imp) == colnames(train_set.i)[p]))!=0){
        var.importance[i,(p-1)]  <- imp[which(row.names(imp) == colnames(train_set.i)[p]),1]
      }
      else{
        var.importance[i,(p-1)]  <- 0
      }
    }
    var.importance.i <- var.importance[i, ]
    
    predict.3 <- predict.train(model3, newdata = test_set.i)
    rmse.results[i, 13] <- Metrics::rmse(predict.3, test_set.i$TI)
    rmse.results[i, 14] <- Metrics::mae(predict.3, test_set.i$TI)
    rmse.results[i, 15] <- rcorr(predict.3, test_set.i$TI)$r[2,1] ^ 2
    predict.3.w <- predict.train(model3, newdata = train_set.i)
    rmse.results[i, 37] <- Metrics::rmse(predict.3.w, train_set.i$TI)
    rmse.results[i, 38] <- Metrics::mae(predict.3.w, train_set.i$TI)
    rmse.results[i, 39] <- rcorr(predict.3.w, train_set.i$TI)$r[2,1] ^ 2
    
    value <- c(rmse.results[i, 7],rmse.results[i, 8],rmse.results[i, 9],
               rmse.results[i, 37],rmse.results[i, 38],rmse.results[i, 39])
    out <- list(value = value,  var.importance = var.importance.i, final.model = model3)
    save(out, file = paste0((m*rep - rep + i), ".RData"))
  }
  rmse.results.list[[m]] <- rmse.results
  var.importance.model3[[m]] <- var.importance
}

########################################################################
## model6: (single) RE-EM tree
## since the training procedure is not yet automatizedb by the package caret, here we develop 
## a novel procedure to execute the training
set.seed(970912)
n.fold <- 10
## cp is the tuning parameter in tree models. Here cp is tuned on a grid [.001, .003, ..., .029, .031]
## the grid of cp is kept constant in two (single) tree models
cp.grid <- seq(.031, .001, by = -.002)
## store results in the training process
rmse.mltree <- matrix(0, ncol = n.fold, nrow = length(cp.grid))
setwd("C:/Users/u1275970/Documents/HR analytics/Models/Mutilevel-Prediction-Models/estimates/mltree")
for (m in 1:n.datasets){
  rmse.results <- rmse.results.list[[m]]
  var.importance <- var.importance.model3[[m]]
  if(m == 1){
    test_set <- test_set.f.r
    train_set <- train_set.f.r    
  }
  if(m == 2){
    test_set <- test_set.b.r
    train_set <- train_set.b.r  
  }
  if(m == 3){
    test_set <- test_set.f.s
    train_set <- train_set.f.s 
  }
  if(m == 4){
    test_set <- test_set.b.s
    train_set <- train_set.b.s
  }
  
  ## the training process. For each repetition, a 10-fold cross validation procedure is run and the average 
  ## prediction performance on the test sets are computed and compared
  
  # rep refer to the number of repetitions (=200)
  for (i in 1:rep){
    train_set.i <- as.data.frame(train_set[[i]])
    test_set.i <- as.data.frame(test_set[[i]])
    
    # n.foldrefers to the number of folds (=10)
    for (l in 1:n.fold){
      test.obs <- which(train_set.i$.folds == l)
      train.set <- train_set.i[-test.obs,]
      test.set <- train_set.i[test.obs,]
      train.set<- within(train.set ,rm(.folds))
      
      # cp.grid consists of the candidate values of the tuning parameter cp
      for (j in 1:length(cp.grid)){
        rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[j], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)
        n <- names(train.set)
        # constitute the formula with regard to the datasets under consideration 
        f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
        model4 <- REEMtree(f, data = train.set, random = ~1|code, tree.control = rpart.ctr)
        # compute the prediction performance for each dataset
        a <- predict(model4, newdata = test.set, id=test.set$code)
        rmse.mltree[j,l] <- Metrics::rmse(a, test.set$TI)
      }
    }
    
    # compute the average prediction performance with regard to each potential value of the tuning parameter
    rmse.sum <- apply(rmse.mltree, 1, mean)
    # select the optimal value of the tuning parameter cp
    opt<-which.min(rmse.sum)
    rpart.ctr <-rpart.control(minsplit=10, cp=cp.grid[opt], maxcompete=4, maxsurrogate=5, xval=10, maxdepth=30)
    
    ## obtain the initial random effects. The initial random effects formulate a rational start for more advanced calculations
    n <- names(train.set)
    f1 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + "), "+(1|code)"))
    model4.app <- lmer(f1,  data = train_set.i)
    r <- ranef(model4.app)$code[,1]
    
    # conditional on the optimal value of cp, train the model with the whole training datasets
    train_set.i<- within(train_set.i ,rm(.folds))
    model4.final <- REEMtree(f, data = train_set.i, random = ~1|code, tree.control = rpart.ctr,
                             initialRandomEffects = r)
    
    # obtain the relative importance metrices
    imp <- model4.final$Tree$variable.importance
    for (p in 3:ncol(train_set.i)){
      if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
        var.importance[i,(p-2)]  <- imp[which(names(imp) == colnames(train_set.i)[p])]
      }
      if (length(which(names(imp) == colnames(train_set.i)[p]))==0){
        var.importance[i,(p-2)]  <- 0
      }
    }

    predict.4 <- predict(model4.final, newdata = test_set.i, id = test_set.i$code)
    predict.4.w <- predict(model4.final, newdata = train_set.i, id = train_set.i$code)

    rmse.results[i, 16] <- Metrics::rmse(predict.4, test_set.i$TI)
    rmse.results[i, 17] <- Metrics::mae(predict.4, test_set.i$TI)
    rmse.results[i, 18] <- rcorr(predict.4, test_set.i$TI)$r[2,1] ^ 2

    rmse.results[i, 40] <- Metrics::rmse(predict.4.w, train_set.i$TI)
    rmse.results[i, 41] <- Metrics::mae(predict.4.w, train_set.i$TI)
    rmse.results[i, 42] <- rcorr(predict.4.w, train_set.i$TI)$r[2,1] ^ 2
    
    value <- c(rmse.results[i, 10],rmse.results[i, 11],rmse.results[i, 12],
               rmse.results[i, 40],rmse.results[i, 41],rmse.results[i, 42])
    out <- list(value = value,  final.model = model3)
    save(out, file = paste0((m*rep - rep + i), ".RData"))
  }
  
  rmse.results.list[[m]] <- rmse.results
  var.importance.model4[[m]] <- var.importance
}
################################################################################
## the following code estimates prediction models with parallel computations
################################################################################
## model3: lasso regression
setwd("~/lasso")
set.seed(970912)
## the grid of the tuning parameter
lambda <- seq(14,0,by=-1)
## number of folders
n.fold <- 10
rep <- 200
no_cores <- 25
## The parallel computation is run with the help of R package "doParallel" and "snow".
## Other ways to run parallel computation are also possible
## the code to open the connection and register the computational cores
c1 <- makePSOCKcluster(no_cores)
registerDoParallel(c1)
foreach(i = 1:rep,
        .packages = c("glmmLasso", "Metrics","caret", "randomForest","Hmisc",
                      "rpart",  "car",  "Matrix", "lattice", "nlme", "FSA", "REEMtree"), .combine=rbind) %dopar%{
                        
                        records <- matrix(0, nrow = n.datasets, ncol = 6)
                        model.records <- list()
                        var.importance.i <- list()
                        
                        for (m in 1:n.datasets){
                          rmse.results <- rmse.results.list[[m]]
                          var.importance.i[[m]] <- var.importance.model4[[m]]
                          
                          if(m == 1){
                            test_set <- test_set.f.r
                            train_set <- train_set.f.r    
                          }
                          if(m == 2){
                            test_set <- test_set.b.r
                            train_set <- train_set.b.r  
                          }
                          if(m == 3){
                            test_set <- test_set.f.s
                            train_set <- train_set.f.s 
                          }
                          if(m == 4){
                            test_set <- test_set.b.s
                            train_set <- train_set.b.s
                          }
                          
                          train_set.i <- train_set[[i]]
                          train_set.i <- within(train_set.i, rm(code))
                          test_set.i <- test_set[[i]]
                          test_set.i <- within(test_set.i ,rm(code))
                          
                          rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
                          
                          ## training procedure with cross validation
                          for (l in 1:n.fold){
                            test.obs <- which(train_set.i$.folds == l)
                            train.set <- train_set.i[-test.obs,]
                            test.set <- train_set.i[test.obs,]
                            train.set<- within(train.set ,rm(.folds))
                            ## starting values of the training parameters
                            delta.start <- as.matrix(t(rep(0, ncol(train.set))))
                            q.start <- .1
                            
                            for (j in 1:length(lambda)){
                              n <- names(train.set)
                              ## construct the formula that is specific to the datasets under consideration
                              f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI")], collapse = " + ")))
                              model5 <- glmmLasso::glmmLasso(f, rnd = NULL, data = train.set, 
                                                             lambda = lambda[j], switch.NR = T, final.re = TRUE,
                                                             control = list(start=delta.start[j,], q_start=q.start))  
                              ## record the prediction performance of the current fold and update the starting values of the parameters
                              rmse.lasso[j,l] <- rmse(predict(model5, newdata = test.set), test.set$TI)
                              delta.start<-rbind(delta.start,model5$Deltamatrix[model5$conv.step,])
                            }
                          }
                          rmse.mean <- apply(rmse.lasso, 1, mean)
                          ## select the optimal value of the tuning parameter
                          opt<- which.min(rmse.mean)
                          train_set.i<- within(train_set.i ,rm(.folds))
                          
                          # conditional on the optimal value of cp, train the model with the whole training datasets
                          model5.final <- glmmLasso(f, rnd = NULL, data = train_set.i, 
                                                    lambda = lambda[opt], switch.NR = T, final.re = TRUE,
                                                    control = list(start=delta.start[opt,], q_start=q.start))
                          
                          # obtain the relative importance metrics of the current analyses
                          imp <- abs(summary(model5.final)$coefficients[,3])
                          for (p in 2:ncol(train_set.i)){
                            if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
                              if (is.na(imp[which(names(imp) == colnames(train_set.i)[p])])){
                                var.importance.i[[m]][p-1] <- 0
                              }
                              else{
                                var.importance.i[[m]][p-1]  <- imp[which(names(imp) == colnames(train_set.i)[p])]        }
                            }
                          }
                          
                          predict.5 <- predict(model5.final, newdata = test_set.i)
                          predict.5.w <- predict(model5.final, newdata = train_set.i)
                          
                          records[m,1] <- Metrics::rmse(predict.5, test_set.i$TI)
                          records[m,2] <- Metrics::mae(predict.5, test_set.i$TI)
                          records[m,3] <- rcorr(predict.5, test_set.i$TI)$r[2,1] ^ 2
                          
                          records[m,4] <- Metrics::rmse(predict.5.w, train_set.i$TI)
                          records[m,5] <- Metrics::mae(predict.5.w, train_set.i$TI)
                          records[m,6] <- rcorr(predict.5.w, train_set.i$TI)$r[2,1] ^ 2
                          
                          model.records[[m]] <- model5.final
                        }
                        
                        out <- list(records = records, var.importance = var.importance.i, final.model = model.records)
                        save(out, file = paste0(i, ".RData"))
                      }
stopCluster(c1)

###########################################################################
## model4:penalized linear mixed model (PLMM)
setwd("~/mllasso")
set.seed(970912)
## to keep the unit-level model and the multilevel model comparable, we set the same grid for the tuning parameter cp
lambda <- seq(14,0,by=-1)
n.fold <- 10
rep <- 200
no_cores <- 25
## the training process is mostly identical to the training process of model3, so we will skip the notations here
c1 <- makePSOCKcluster(no_cores)
registerDoParallel(c1)
foreach(i = 1:rep,
        .packages = c("glmmLasso", "Metrics","caret", "randomForest","Hmisc",
                      "rpart",  "car",  "Matrix", "lattice", "nlme", "FSA", "REEMtree"), .combine=rbind) %dopar%{
                        
                        records <- matrix(0, nrow = n.datasets, ncol = 6)
                        model.records <- list()
                        var.importance.i <- list()
                        
                        for (m in 1:n.datasets){
                          rmse.results <- rmse.results.list[[m]]
                          var.importance.i[[m]] <- var.importance.model3[[m]]
                          if(m == 1){
                            test_set <- test_set.f.r
                            train_set <- train_set.f.r    
                          }
                          if(m == 2){
                            test_set <- test_set.b.r
                            train_set <- train_set.b.r  
                          }
                          if(m == 3){
                            test_set <- test_set.f.s
                            train_set <- train_set.f.s 
                          }
                          if(m == 4){
                            test_set <- test_set.b.s
                            train_set <- train_set.b.s
                          }

                          train_set.i <- as.data.frame(train_set[[i]])
                          test_set.i <- as.data.frame(test_set[[i]])

                          rmse.lasso <- matrix(0, ncol = n.fold, nrow = length(lambda))
                          
                          for (l in 1:n.fold){
                            test.obs <- which(train_set.i$.folds == l)
                            train.set <- train_set.i[-test.obs,]
                            test.set <- train_set.i[test.obs,]
                            train.set<- within(train.set ,rm(.folds))
                            delta.start <- as.matrix(t(rep(0, ncol(train.set) - 1 + length(levels(train.set$code)))))
                            q.start <- .1
                            n <- names(train.set)
                            f <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
                            for (j in 1:length(lambda)){
                              model6 <- glmmLasso(f, rnd = list(code =~ 1), data = train.set, 
                                                  lambda = lambda[j], switch.NR = F, final.re = TRUE,
                                                  control = list(start=delta.start[j,], q_start=q.start[j]))  
                              rmse.lasso[j,l] <- rmse(predict(model6, newdata = test.set), test.set$TI)
                              delta.start<-rbind(delta.start,model6$Deltamatrix[model6$conv.step,])
                              q.start<-c(q.start,model6$Q_long[[model6$conv.step+1]])
                            }
                          }
                          
                          rmse.mean <- apply(rmse.lasso, 1, mean)
                          opt <- which.min(rmse.mean)
                          train_set.i <- within(train_set.i ,rm(.folds))
                          model6.final <- glmmLasso(f, rnd = list(code =~ 1), data = train_set.i, 
                                                    lambda = lambda[opt], switch.NR = F, final.re = TRUE,
                                                    control = list(start=delta.start[opt,], q_start=q.start[opt])) 
                          
                          imp <- abs(summary(model6.final)$coefficients[,3])
                          for (p in 3:ncol(train_set.i)){
                            if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
                              if (is.na(imp[which(names(imp) == colnames(train_set.i)[p])])){
                                var.importance.i[[m]][p-2] <- 0
                              }
                              else{
                                var.importance.i[[m]][p-2]  <- imp[which(names(imp) == colnames(train_set.i)[p])]
                              }
                            }
                          }
                          
                          predict.6 <- predict(model6.final, newdata = test_set.i)
                          predict.6.w <- predict(model6.final, newdata = train_set.i)
                          
                          records[m,1] <- Metrics::rmse(predict.6, test_set.i$TI)
                          records[m,2] <- Metrics::mae(predict.6, test_set.i$TI)
                          records[m,3] <- rcorr(predict.6, test_set.i$TI)$r[2,1] ^ 2
                          
                          records[m,4] <- Metrics::rmse(predict.6.w, train_set.i$TI)
                          records[m,5] <- Metrics::mae(predict.6.w, train_set.i$TI)
                          records[m,6] <- rcorr(predict.6.w, train_set.i$TI)$r[2,1] ^ 2
                          
                          model.records[[m]] <- model6.final
                        }
                        
                        out <- list(records = records, var.importance = var.importance.i, final.model = model.records)
                        save(out, file = paste0(i, ".RData"))
                      }

stopCluster(c1)

###########################################################################
## model 7: bagged trees
setwd("~/bt")
set.seed(970912)
## Each tree in bagged trees is supposed to grow as large as possible (i.e. without active pruning)
## therefore, the value of the tuning parameter cp is fixed at a minimal value .01
## no cross validation procedure is incolved in bagged trees and bagged RE-EM trees
cp.grid <- 0.01
## values of other parameters 
n.fold <- 10
n.boot <- 500
rep <- 200
no_cores <- 25
## open the connection and register the cores
c1 <- makePSOCKcluster(no_cores)
registerDoParallel(c1)
foreach(i = 1:rep,
        .packages = c("glmmLasso", "Metrics","caret", "randomForest","Hmisc",
                      "rpart",  "car",  "Matrix", "lattice", "nlme", "FSA", "REEMtree"), .combine=rbind) %dopar%{
                        
                        records <- matrix(0, nrow = n.datasets, ncol = 6)
                        model.records <- list()
                        var.importance.i <- list()
                        var.importance.i.sum <- list()
                        
                        for (m in 1:n.datasets){
                          if(m == 1){
                            train_set <- train_set.f.r
                            test_set <- test_set.f.r
                            var.importance.i[[1]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 3)
                            var.importance.i.sum[[1]] <- vector(length = ncol(train_set[[1]]) - 3)
                          }

                          if(m == 2){
                            train_set <- train_set.b.r
                            test_set <- test_set.b.r
                            var.importance.i[[2]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 3)
                            var.importance.i.sum[[2]] <- vector(length = ncol(train_set[[1]]) - 3)
                          }

                          if(m == 3){
                            train_set <- train_set.f.s
                            test_set <- test_set.f.s
                            var.importance.i[[3]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 3)
                            var.importance.i.sum[[3]] <- vector(length = ncol(train_set[[1]]) - 3)
                          }

                          if(m == 4){
                            train_set <- train_set.b.s
                            test_set <- test_set.b.s
                            var.importance.i[[4]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 3)
                            var.importance.i.sum[[4]] <- vector(length = ncol(train_set[[1]]) - 3)
                          }
                          
                          train_set.i <- within(train_set[[i]], rm(code))
                          test_set.i <- within(test_set[[i]], rm(code))
                          
                          # note that the training procedure is without cross validation 
                          my_control <- trainControl(method = "none", 
                                                     number = 10, 
                                                     verboseIter = FALSE,
                                                     allowParallel = FALSE)
                          
                          bag.pred <- list()
                          bag.pred.in <- list()
                          train_set.i <- within(train_set.i, rm(.folds))
                          
                          # to make each bootsrtapped tree, we first create a bootstrap sample by randomly sampling (with replacement) 
                          # from the original sample so that the size of the bootstrap sample equals the size of the original sample. Then,
                          # we create none-pruned trees for each bootstrap sample. 
                          for (x in 1:n.boot){
                            train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
                            model7 <- caret::train(TI ~ ., data = train_set.x, 
                                                   method = "rpart", 
                                                   trControl = my_control,
                                                   tuneGrid = expand.grid(cp = cp.grid),
                                                   metric = "RMSE",
                                                   # note that the other parameters specified in the tree models are also 
                                                   # determined in a way that minizes the amount of pruning
                                                   control = rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30))
                            # obtain the relative importance metrics of each bootstrapped tree
                            imp <- varImp(model7)$importance
                            for (p in 2:ncol(train_set.i)){
                              if (length(which(row.names(imp) == colnames(train_set.i)[p]))!=0){
                                var.importance.i[[m]][x,(p-1)]  <- imp[which(row.names(imp) == colnames(train_set.i)[p]),1]
                              }
                              else{
                                var.importance.i[[m]][x,(p-1)]  <- 0
                              }
                            }
                            # generate the prediction (for the validation set) based on each bootstrapped tree
                            bag.pred[[x]] <- predict(model7, newdata = test_set.i)
                            # generate the prediction (for the training set) based on each bootstrapped tree
                            bag.pred.in[[x]] <- predict(model7, newdata = train_set.i)
                          }
                          
                          # the final prediction equals the average of all predictions derived from all bootstrapped trees
                          predict.7 <- Reduce("+", bag.pred)/n.boot
                          predict.7.w <- Reduce("+", bag.pred.in)/n.boot
                          
                          records[m,1] <- Metrics::rmse(predict.7, test_set.i$TI)
                          records[m,2] <- Metrics::mae(predict.7, test_set.i$TI)
                          records[m,3] <- rcorr(predict.7, test_set.i$TI)$r[2,1] ^ 2
                          
                          records[m,4] <- Metrics::rmse(predict.7.w, train_set.i$TI)
                          records[m,5] <- Metrics::mae(predict.7.w, train_set.i$TI)
                          records[m,6] <- rcorr(predict.7.w, train_set.i$TI)$r[2,1] ^ 2
                          
                          var.importance.i.sum[[m]] <- apply(var.importance.i[[m]], 2, mean)
                        }
                        
                        out <- list(records = records, var.importance = var.importance.i.sum, final.pred = predict.7, final.pred.in = predict.7.w)
                        save(out, file = paste0(i, ".RData"))
                      }

stopCluster(c1)



###########################################################################
## model 8: random effects RE-EM trees
## the very similar training procedure is applied to model 8 and model 7.
## therefore, the annotations will not be repeated here. 
setwd("~/mlbt")
index <- vector()
set.seed(970912)
# the same idea applied here: the bootstrapped trees are grown without pruning 
cp.grid <- .01
n.fold <- 10
n.boot <- 500
no_cores <- 25
rep <- 200
## make the connection and register the computational cores
c1 <- makePSOCKcluster(no_cores)
registerDoParallel(c1)
rmse.lasso <- foreach(i = index,
                      .packages = c("glmmLasso", "Metrics","caret", "randomForest","Hmisc","lme4",
                                    "rpart",  "car",  "Matrix", "lattice", "nlme", "FSA", "REEMtree"), .combine=rbind) %dopar%{
                                      
                                      records <- matrix(0, nrow = n.datasets, ncol = 6)
                                      model.records <- list()
                                      var.importance.i <- list()
                                      var.importance.i.sum <- list()
                                      
                                      for (m in 1:n.datasets){
                                        if(m == 1){
                                          train_set <- train_set.f.r
                                          test_set <- test_set.f.r
                                          var.importance.i[[1]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 2)
                                          var.importance.i.sum[[1]] <- vector(length = ncol(train_set[[1]]) - 2)
                                        }
                                        if(m == 2){
                                          train_set <- train_set.b.r
                                          test_set <- test_set.b.r
                                          var.importance.i[[2]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 2)
                                          var.importance.i.sum[[2]] <- vector(length = ncol(train_set[[1]]) - 2)
                                        }
                                        if(m == 3){
                                          train_set <- train_set.f.s
                                          test_set <- test_set.f.s
                                          var.importance.i[[3]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 2)
                                          var.importance.i.sum[[3]] <- vector(length = ncol(train_set[[1]]) - 2)
                                        }
                                        if(m == 4){
                                          train_set <- train_set.b.s
                                          test_set <- test_set.b.s
                                          var.importance.i[[4]] <- matrix(nrow = n.boot, ncol = ncol(train_set[[1]]) - 2)
                                          var.importance.i.sum[[4]] <- vector(length = ncol(train_set[[1]]) - 2)
                                        }
                                        
                                        train_set.i <- as.data.frame(train_set[[i]])
                                        test_set.i <- as.data.frame(test_set[[i]])
                                        train_set.i <- within(train_set.i, rm(.folds))
                                        
                                        bag.pred <- list()
                                        bag.pred.in <- list()
                                        
                                        for (x in 1:n.boot){
                                          train_set.x <- train_set.i[base::sample(1:nrow(train_set.i), size = nrow(train_set.i), replace = TRUE),]
                                          rpart.ctr <-rpart.control(minsplit = 10, minbucket = 3, maxdepth = 30, cp = cp.grid)
                                          
                                          train_set.x$code <- factor(train_set.x$code)
                                          n <- names(train_set.x)
                                          f2 <- as.formula(paste("TI ~", paste(n[!n %in% c("TI", "code")], collapse = " + ")))
                                          
                                          model8.final <- REEMtree(f2, data = train_set.x, random = ~1|code, tree.control = rpart.ctr)
                                          
                                          
                                          bag.pred[[x]] <- predict(model8.final, newdata = test_set.i, id = test_set.i$code)
                                          bag.pred.in[[x]] <- predict(model8.final, newdata = train_set.i, id = train_set.i$code)
                                          
                                          
                                          imp <- model8.final$Tree$variable.importance
                                          for (p in 3:ncol(train_set.i)){
                                            if (length(which(names(imp) == colnames(train_set.i)[p]))!=0){
                                              var.importance.i[[m]][x,(p-2)]  <- imp[which(names(imp) == colnames(train_set.i)[p])]
                                            }
                                            if (length(which(names(imp) == colnames(train_set.i)[p]))==0){
                                              var.importance.i[[m]][x,(p-2)]  <- 0
                                            }
                                          }
                                        }
                                        
                                        predict.8 <- Reduce("+", bag.pred)/n.boot
                                        predict.8.w <- Reduce("+", bag.pred.in)/n.boot
                                        
                                        records[m,1] <- Metrics::rmse(predict.8, test_set.i$TI)
                                        records[m,2] <- Metrics::mae(predict.8, test_set.i$TI)
                                        records[m,3] <- rcorr(predict.8, test_set.i$TI)$r[2,1] ^ 2
                                        
                                        records[m,4] <- Metrics::rmse(predict.8.w, train_set.i$TI)
                                        records[m,5] <- Metrics::mae(predict.8.w, train_set.i$TI)
                                        records[m,6] <- rcorr(predict.8.w, train_set.i$TI)$r[2,1] ^ 2
                                        
                                        var.importance.i.sum[[m]] <- apply(var.importance.i[[m]], 2, mean)
                                      }
                                      
                                      out <- list(records = records, var.importance = var.importance.i.sum, final.pred = predict.8, final.pred.in = predict.8.w)
                                      save(out, file = paste0(i, ".RData"))
                                    }

stopCluster(c1)


#########################################################################################
# part3: data summarization and figure construction ( only for review purpose)
#########################################################################################
## the resilts obtained from estimation without parallel computation should be stored locally,
## while the results obtain from estimation with parallebel computation is summarized in one RData file
load("simresults.RData")
rmse.results.supplement <- sim.results[[1]]
var.importance.supplement <- sim.results[[2]]
####### summarize results #####################
## some of the results (obtained from estimation without parallel computation) has been already been 
## stores in rmse.results.list, therefore does not need to be read in again
## results that represent prediction performance 
for (i in 1:n.datasets){
  rmse.results.list[[i]][, 7:9] <- rmse.results.supplement[[i]][,1:3]
  rmse.results.list[[i]][, 31:33] <- rmse.results.supplement[[i]][,4:6]
  rmse.results.list[[i]][, 10:12] <- rmse.results.supplement[[i]][,7:9]
  rmse.results.list[[i]][, 34:36] <- rmse.results.supplement[[i]][,10:12]
  rmse.results.list[[i]][, 19:21] <- rmse.results.supplement[[i]][,13:15]
  rmse.results.list[[i]][, 43:45] <- rmse.results.supplement[[i]][,16:18]
  rmse.results.list[[i]][, 22:24] <- rmse.results.supplement[[i]][,19:21]
  rmse.results.list[[i]][, 46:48] <- rmse.results.supplement[[i]][,22:24]
}

## results that indicate relative importance of the variables
var.importance.all <- list()
var.importance.all[[1]] <- var.importance.model1[[2]]
var.importance.all[[2]] <- var.importance.model2[[2]]
var.importance.all[[3]] <- var.importance.supplement[[1]]
var.importance.all[[4]] <- var.importance.supplement[[2]]
var.importance.all[[5]] <- var.importance.supplement[[3]]
var.importance.all[[6]] <- var.importance.supplement[[4]]

## process the data and coerce the results into a data frame
rmse.results.r2 <- list()
rmse.results.rmse <- list()
col.names <- rep(c("reg", "rereg", "preg", "prereg", "tree", "retree", "btree", "bretree"), 2)
for (i in 1:n.datasets){
  ## only retain the r square
  rmse.results.r2[[i]] <- rmse.results.list[[i]][,seq(3,48,by = 3)]
  rmse.results.rmse[[i]] <- rmse.results.list[[i]][, seq(1,46,by = 3)]
  colnames(rmse.results.r2[[i]]) <- col.names
  colnames(rmse.results.rmse[[i]]) <- col.names
}
## data processing
rmse.r2.dataset <- data.frame()
rmse.rmse.dataset <- data.frame()
for(i in 1:n.datasets){
  if(i == 1){
    rmse.results.r2.i <- as.data.frame(rmse.results.r2[[i]])
    rmse.results.rmse.i <- as.data.frame(rmse.results.rmse[[i]])
    rmse.results.r2.i$level <- rep("individual", nrow(rmse.results.r2.i))
    rmse.results.r2.i$prediction <- rep("new_group", nrow(rmse.results.r2.i))
    rmse.results.rmse.i$level <- rep("individual", nrow(rmse.results.rmse.i))
    rmse.results.rmse.i$prediction <- rep("new_group", nrow(rmse.results.rmse.i))
  }
  if(i == 2){
    rmse.results.r2.i <- as.data.frame(rmse.results.r2[[i]])
    rmse.results.rmse.i <- as.data.frame(rmse.results.rmse[[i]])
    rmse.results.r2.i$level <- rep("both", nrow(rmse.results.r2.i))
    rmse.results.r2.i$prediction <- rep("new_group", nrow(rmse.results.r2.i))
    rmse.results.rmse.i$level <- rep("both", nrow(rmse.results.rmse.i))
    rmse.results.rmse.i$prediction <- rep("new_group", nrow(rmse.results.rmse.i))
  }
  if(i == 3){
    rmse.results.r2.i <- as.data.frame(rmse.results.r2[[i]])
    rmse.results.rmse.i <- as.data.frame(rmse.results.rmse[[i]])
    rmse.results.r2.i$level <- rep("individual", nrow(rmse.results.r2.i))
    rmse.results.r2.i$prediction <- rep("old_group", nrow(rmse.results.r2.i))
    rmse.results.rmse.i$level <- rep("individual", nrow(rmse.results.rmse.i))
    rmse.results.rmse.i$prediction <- rep("old_group", nrow(rmse.results.rmse.i))
  }
  if(i == 4){
    rmse.results.r2.i <- as.data.frame(rmse.results.r2[[i]])
    rmse.results.rmse.i <- as.data.frame(rmse.results.rmse[[i]])
    rmse.results.r2.i$level <- rep("both", nrow(rmse.results.r2.i))
    rmse.results.r2.i$prediction <- rep("old_group", nrow(rmse.results.r2.i))
    rmse.results.rmse.i$level <- rep("both", nrow(rmse.results.rmse.i))
    rmse.results.rmse.i$prediction <- rep("old_group", nrow(rmse.results.rmse.i))
  }
  rmse.r2.dataset <- rbind(rmse.r2.dataset, rmse.results.r2.i)
  rmse.rmse.dataset <- rbind(rmse.rmse.dataset, rmse.results.rmse.i)  
}

rmse.r2.new <- rmse.r2.dataset[,c(1:8, 17:18)]
rmse.r2.old <- rmse.r2.dataset[,9:17]
rmse.rmse.new <- rmse.rmse.dataset[,c(1:8, 17:18)]
rmse.rmse.old <- rmse.rmse.dataset[,9:17]

##############################################
## the following code is used to make the plots that indicate out-of-sample prediction performance  
method_names <- c("reg", "rereg", "preg", "prereg", "tree", "retree", "btree", "bretree")
summary.r2 <- tidyr::gather(rmse.r2.new, methods, r_square, method_names, factor_key=TRUE)
summary.r2$methods <- as.factor(summary.r2$methods)
levels(summary.condition1$methods) <- c("CardKmeans", "Kmeans", "GMM")
summary.r2 %>%
  ggplot(aes(x=prediction, y=r_square, fill=methods)) + 
  geom_boxplot() +
  facet_grid(level~prediction, scale="free_x", labeller = labeller(
    level = c('both' = "predictors from both levels", 'individual' = "predictors from individual level"),
    prediction = c('new_group' = "observations from new groups",
                   'old_group' = "observations from existing groups")
  )) +
  labs(x = "Types of Prediction", y = "R Square") +
  theme(axis.title.x = element_text(size = 14, face = "bold"), axis.title.y = element_text(size = 14, face = "bold"),
        strip.text.x = element_text(size = 12.5, face = "bold"), strip.text.y = element_text(size = 12.5, face = "bold"),
        legend.title = element_text(size = 13, face = "bold"),
        axis.text.x = element_blank(),
        legend.text = element_text(size = 13, face = "bold"))
## the following code is used to make the plots that indicate in-sample prediction performance  
method_names <- c("reg", "rereg", "preg", "prereg", "tree", "retree", "btree", "bretree")
summary.r2 <- tidyr::gather(rmse.r2.old, methods, r_square, method_names, factor_key=TRUE)
summary.r2$methods <- as.factor(summary.r2$methods)
levels(summary.condition1$methods) <- c("CardKmeans", "Kmeans", "GMM")
summary.r2 %>%
  ggplot(aes(x=level, y=r_square, fill=methods)) + 
  geom_boxplot() +
  facet_grid(~level, scale="free_x", labeller = labeller(
    level = c('both' = "predictors from both levels", 'individual' = "predictors from individual level")
  )) +
  labs(x = "Levels of Predictors", y = "R Square") +
  theme(axis.title.x = element_text(size = 14, face = "bold"), axis.title.y = element_text(size = 14, face = "bold"),
        strip.text.x = element_text(size = 12.5, face = "bold"), strip.text.y = element_text(size = 12.5, face = "bold"),
        legend.title = element_text(size = 13, face = "bold"),
        axis.text.x = element_blank(),
        legend.text = element_text(size = 13, face = "bold"))

## summarize the data to extract some discrete statistics to report
rmse.r2.new %>%
  dplyr::filter(level == "both") %>%
  dplyr::summarise(tree.sum = mean(tree), retree.sum = mean(retree), btree.sum = mean(btree), bretree.sum = mean(bretree))
rmse.r2.new %>%
  dplyr::filter(level == "individual") %>%
  dplyr::summarise(tree.sum = mean(tree), retree.sum = mean(retree), btree.sum = mean(btree), bretree.sum = mean(bretree))
a <- rmse.r2.new %>%
  dplyr::filter(level == "individual") %>%
  dplyr::summarise(reg.sum = mean(reg), rereg.sum = mean(rereg), preg.sum = mean(preg), prereg.sum = mean(prereg))
b <- rmse.r2.new %>%
  dplyr::filter(prediction == "old_group") %>%
  dplyr::group_by(level) %>%
  dplyr::summarise(tree.sum = mean(tree), retree.sum = mean(retree), btree.sum = mean(btree), bretree.sum = mean(bretree),
                   reg.sum = mean(reg), rereg.sum = mean(rereg), preg.sum = mean(preg), prereg.sum = mean(prereg))
c <- rmse.r2.old %>%
  dplyr::group_by(level) %>%
  dplyr::summarise(tree.sum = mean(tree), retree.sum = mean(retree), btree.sum = mean(btree), bretree.sum = mean(bretree),
                   reg.sum = mean(reg), rereg.sum = mean(rereg), preg.sum = mean(preg), prereg.sum = mean(prereg))


####################
##the following code is used to summarized the average variable importance indices across various methods and various datasets into a table 
var.importance <- as.data.frame(matrix(nrow = ncol(var.importance.all[[1]]), ncol = 6))
for (i in 1:6){
  var.importance[,i] <- apply(var.importance.all[[i]], 2, mean)
}
var.names <- colnames(var.importance.all[[1]])
row.names(var.importance) <- var.names
method_names1 <- c("reg", "rereg", "preg", "prereg",  "btree", "bretree")
colnames(var.importance) <- method_names1
for (i in 1:6){
  var.importance[,i] <- scale(var.importance[,i])
}
var.importance.vector <- apply(var.importance,1,mean)
sort.rows <- sort.int(var.importance.vector, index.return = TRUE, decreasing = TRUE)$ix
var.importance.vector[sort.rows]

var.importance.content <- list(var.importance, var.importance.all, var.importance.vector)
save(var.importance.content, file = "importance.RData")

#####################################################################################
## end of the code ##################################################################
#####################################################################################

