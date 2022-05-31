library(tidyverse)
library(ggplot2)
library(gridExtra)
library(sparseLDA)
library(MASS)
library(nnet)
library(caret)
library(plotly)
library(stats)
library(glmnet)
library(plotROC)
library(pROC)
library(cowplot)
library(dplyr)
library(tidyr)
library(purrr)
library(tidymodels)
library(scutr)

##### load DATA #########################
# tot <- read.csv(file.choose(), header = FALSE)
df1 <- read.csv('data/SFinGe_Default.csv', header = FALSE)
df1$quality <- 'default' 

df2 <- read.csv('data/SFinGe_HQNoPert.csv', header = FALSE)
df2$quality <- 'hq' 

df3 <- read.csv('data/SFinGe_VQAndPert.csv', header = FALSE)
df3$quality <- 'vq' 

tot <- rbind(df1,df2,df3)
colnames(tot)[ncol(tot)-1] <- 'label'
tot$quality <- as.factor(tot$quality)
tot$label <- as.factor(tot$label)
table(tot$quality, tot$label)
table(tot$label)

dim(tot)
sum(is.na(tot))

tot$label <- as.factor(tot$label)
tot$quality <- as.factor(tot$quality)
# tot[,1:530] %>%
#   glimpse()

## Glimpse first 10 columns
head(tot[, 1:10], n = 14)

# # #### Remove Outliers ######
cap <- function(x){
  quantiles <- quantile( x, c(.05, 0.25, 0.75, .95 ) )
  x[ x < quantiles[2] - 1.5*IQR(x) ] <- quantiles[1]
  x[ x > quantiles[3] + 1.5*IQR(x) ] <- quantiles[4]
  x}

tot <- tot %>% mutate_if(is.numeric, cap)

boxplot(tot[, 1:6])

# Finding if columns have Zero Variance that gives NAs while scaling
x <- cbind(lapply(tot[,1:(ncol(tot)-2)], FUN = var, na.rm = T))

vardf <- data.frame('col' = rownames(x), 'variation' = unlist(x))
vardf$col[round(vardf$variation, 4) == 0.000]
zero_var <- vardf[order(vardf$variation),]
str(tot$V1312)
label <- tot$label
quality <- tot$quality
# remove columns with zero variance
tot  <- tot[,!(round(vardf$variation, 4) == 0.000)]
tot$label <- label
tot$quality <- quality
dim(tot)


boxplot(tot[, 1:6])

tot <- (tot[, sapply(tot, function(x) length(unique(x)) > 3)[1:(ncol(tot)-2)]])
dim(tot)

###### CLASS BALANCING #####
prop.table(table(tot$label))
tot_de <- tot[tot$quality == 'default',]
tot_hq <- tot[tot$quality == 'hq',]
tot_vq <- tot[tot$quality == 'vq',]
dim(tot_de)

## SMOTE for class A
smoted_deA <- oversample_smote((tot_de %>% select(-quality)), "A", 'label', 700)
smoted_hqA <- oversample_smote((tot_hq %>% select(-quality)), "A", 'label', 700)
smoted_vqA <- oversample_smote((tot_vq %>% select(-quality)), "A", 'label', 700)

smoted_deA$quality <- 'default'
smoted_hqA$quality <- 'hq'
smoted_vqA$quality <- 'vq'

tot <- rbind(tot, smoted_deA, smoted_hqA, smoted_vqA)

## SMOTE for class T
smoted_deT <- oversample_smote((tot_de %>% select(-quality)), "T", 'label', 700)
smoted_hqT <- oversample_smote((tot_hq %>% select(-quality)), "T", 'label', 700)
smoted_vqT <- oversample_smote((tot_vq %>% select(-quality)), "T", 'label', 700)

smoted_deT$quality <- 'default'
smoted_hqT$quality <- 'hq'
smoted_vqT$quality <- 'vq'

tot <- rbind(tot, smoted_deT, smoted_hqT, smoted_vqT)

plot(prop.table(table(tot$label)))

dim(tot)
sum(is.na(tot))
tot %>% glimpse()

# saving the pre processed SFinGe data 
# saveRDS(tot, 'processed_SFinGe.rds')
tot <- readRDS('processed_SFinGe.rds')

####### Train Test Split scale for PCA #################
set.seed(2022)
split <- tot %>%
  # mutate_if(is.numeric, scale) %>%
  initial_split(prop = 7/10, strata = label) # Stratified Sampling
train <- training(split)
test <- testing(split)
Xtrain <- model.matrix( ~ ., dplyr::select(train, -label))[, -1]
Ytrain <- train$label

Xtest <- model.matrix( ~ ., dplyr::select(test, -label))[, -1]
Ytest <- test$label

dim(Xtrain)
length(Ytrain)
dim(Xtest)
length(Ytrain)

table(Ytrain)
table(Ytest)


Xtrain <- scale(Xtrain, center = TRUE, scale = TRUE)
Xtest <- scale(Xtest, center = TRUE, scale = TRUE)
sum(is.na(Xtrain))
sum(is.na(Xtest))


#### Scaling X for PCA #############
# X <- scale(X, center = TRUE, scale = TRUE)
sum(is.na(Xtrain))
svdX <- svd(Xtrain)

nX <- nrow(Xtrain)
r <- ncol(svdX$v)

totVar <- sum(svdX$d^2)/(nX-1)
vars <- data.frame(comp=1:r,var=svdX$d^2/(nX-1)) %>%
  mutate(propVar=var/totVar,cumVar=cumsum(var/totVar))

pVar2 <- vars %>%
  ggplot(aes(x=comp:r,y=propVar)) +
  geom_point() +
  geom_line() +
  xlab("PC") +
  ylab("Proportion of Total Variance")

pVar3 <- vars %>%
  ggplot(aes(x=comp:r,y=cumVar)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = 0.90, color='red')+
  xlab("PC") +
  ylab("Cumulative Proportion of Total Variance")

grid.arrange(pVar2, pVar3, nrow=1)


k <- 310
Vk <- svdX$v[,1:k]
Uk <- svdX$u[,1:k]
Dk <- diag(svdX$d[1:k])
Zk <- Uk%*%Dk
colnames(Zk) <- paste0("PC",1:k)


Zk %>%
  as.data.frame %>%
  mutate(Label = Ytrain %>% as.factor) %>%
  ggplot(aes(x= PC1, y = PC2, color = Label)) +
  geom_point(size = 3)
######################## PC plots ########################################################
axis = list(showline=FALSE,
            zeroline=FALSE,
            gridcolor='#ffff',
            ticklen=4)

fig <- Zk %>%
  as.data.frame %>%
  mutate(Label = Ytrain %>% as.factor) %>%
  plot_ly() %>%
  add_trace(
    type = 'splom',
    dimensions = list(
      list(label='PC1', values=~PC1),
      list(label='PC2', values=~PC2),
      list(label='PC3', values=~PC3),
      list(label='PC4', values=~PC4)
    ),
    color=~Label,
    marker = list(
      size = 7
    )
  ) %>% style(diagonal = list(visible = F)) %>%
  layout(
    title= 'tit',
    hovermode='closest',
    dragmode= 'select',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=1),
    yaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=1),
    xaxis2=axis,
    xaxis3=axis,
    xaxis4=axis,
    yaxis2=axis,
    yaxis3=axis,
    yaxis4=axis
  )
options(warn=-1)
fig

####################### 3D plot ######################################################


fig <- plot_ly(Zk %>%
                 as.data.frame %>%
                 mutate(Label = Ytrain %>% as.factor), x = ~PC1, y = ~PC2, z = ~PC3, color = ~Ytrain, colors = c('#636EFA','#EF553B','#00CC96') ) %>%
  add_markers(size = 12)


fig <- fig %>%
  layout(
    title = 'PC 1, 2 and 3 dimensions',
    scene = list(bgcolor = "#e5ecf6")
  )

fig

# #################### PC loadings #####################
par(mfrow = c(1, 2))

hist(Vk[, 1], breaks = 50, xlab = "PC 1 loadings", main = "")
abline(v = c(
  quantile(Vk[, 1], 0.05),
  quantile(Vk[, 1], 0.95)), col = "red", lwd = 2)


hist(Vk[, 9], breaks = 50, xlab = "PC 2 loadings", main = "")
abline(v = c(
  quantile(Vk[, 2], 0.05),
  quantile(Vk[, 2], 0.95)
), col = "red", lwd = 2)


######### lda  ##########################
sf_lda <- lda(x = Xtrain, grouping = Ytrain)

# cols <- c("n" = "red", "t" = "blue")
Vlda <- sf_lda$scaling
Zlda <- Xtrain %*% Vlda
par(mfrow = c(1, 1))

boxplot(Zlda ~ Ytrain, ylab = expression("Z"[1]),
        main = "Separation of finger print type by LDA")



#### PCA SPLIT #####
pca_X <- prcomp(Xtrain)
pca_X_test <- prcomp(Xtest)
pca_var <- pca_X$sdev^2
vars <- pca_var/sum(pca_var)
nX <- nrow(X)
totVar <- sum(pca_X$sdev^2)/(nX-1)
vars <- data.frame(comp=1:ncol(Xtrain),var=pca_X$sdev^2/(nX-1)) %>%
  mutate(propVar=var/totVar,cumVar=cumsum(var/totVar))

pVar2 <- vars %>%
  ggplot(aes(x=comp:ncol(Xtrain),y=propVar)) +
  geom_point() +
  geom_line() +
  xlab("PC") +
  ylab("Proportion of Total Variance")

pVar3 <- vars %>%
  ggplot(aes(x=comp:ncol(Xtrain),y=cumVar)) +
  geom_point() +
  geom_line() +
  geom_hline(yintercept = 0.90, color='red')+
  xlab("PC") +
  ylab("Cumulative Proportion of Total Variance")

grid.arrange(pVar2, pVar3, nrow=1)


# ############ PCA LASSO #############
npc <- 346
Z_train <- pca_X$x[,1:npc]
Z_test <- pca_X_test$x[,1:npc]

# cv_LASSO_pca <- cv.glmnet(
#   x = Z_train,
#   y = Ytrain,
#   alpha = 1,               # Lasso: alpha = 1
#   type.measure = "class",
#   family = "multinomial", type.multinomial = "grouped")
# saveRDS(cv_LASSO_pca, ''models/SFinGe/cv_lasso_pca_model.rds')
cv_LASSO_pca <- readRDS('models/SFinGe/cv_lasso_pca_model.rds')
plot(cv_LASSO_pca)
# 
# lasso_pca <- glmnet(
#   x = Z_train,
#   y = Ytrain,
#   lambda = cv_LASSO_pca$lambda.min,
#   alpha = 1,         #alpha = 1 forLASSO
#   family="multinomial", type.multinomial = "grouped")
# saveRDS(lasso_pca, 'models/SFinGe/lasso_pca_model.rds')
lasso_pca <- readRDS('models/SFinGe/lasso_pca_model.rds')

dim(Z_train)
dim(Z_test)
dfLassoOpt_pca <- data.frame(
  pred = predict(lasso_pca,
                 newx = as.matrix(Z_test),
                 s = cv_lasso_pca$lambda.min,
                 type = "class") %>% c(.),
  known.truth = Ytest)
pi_pca = predict(lasso_pca,
                 newx = as.matrix(Z_test),
                 s = cv_lasso_pca$lambda.min,
                 type = "response")

confusionMatrix(as.factor(dfLassoOpt_pca$pred), as.factor(Ytest))

table(as.factor(dfLassoOpt_pca$pred), as.factor(Ytest))

pi_pca <- as.matrix(as.data.frame(pi_pca))
colnames(pi_pca) <- c('A', 'L', 'R', 'T', 'W')

auc_pca <- multiclass.roc(as.factor(Ytest), pi_pca, levels=c('A', 'L', 'R', 'T', 'W'))
rs_pca <- auc_pca[['rocs']]

plot.roc(rs_pca[[1]][[1]], levels=c('A', 'L', 'R', 'T', 'W'))
sapply(2:length(rs_pca),function(i) lines.roc(rs_pca[[i]][[1]],col=i))

cbind(Ytest,as.data.frame(pi_pca)) %>% roc_curve(Ytest, A:W) %>% autoplot()


# ############ PCA RIDGE #############

dim(Z_train)
dim(Z_test)

# cv_ridge_pca <- cv.glmnet(
#   x = Z_train,
#   y = Ytrain,
#   alpha = 0,               # ridge: alpha = 0
#   type.measure = "class",
#   family = "multinomial", type.multinomial = "grouped")
# saveRDS(cv_ridge_pca, 'models/SFinGe/ridge_PCA_model.rds')
cv_ridge_pca <- readRDS('models/SFinGe/ridge_PCA_model.rds')

plot(cv_ridge_pca)

# ridge_pca <- glmnet(
#   x = Z_train,
#   y = Ytrain,
#   lambda = cv_ridge_pca$lambda.min,
#   alpha = 0,         #alpha = 0 forL ridge
#   family="multinomial", type.multinomial = "grouped")
# saveRDS(ridge_pca, 'models/SFinGe/ridge_PCA.rds')
ridge_pca <- readRDS('models/SFinGe/ridge_PCA.rds')

dfRidgeOpt_pca <- data.frame(
  pred = predict(ridge_pca,
                 newx = as.matrix(Z_test),
                 s = cv_ridge_pca$lambda.min,
                 type = "class") %>% c(.),
  known.truth = Ytest)
pi_pca = predict(ridge_pca,
                 newx = as.matrix(Z_test),
                 s = cv_ridge_pca$lambda.min,
                 type = "response")

confusionMatrix(as.factor(dfRidgeOpt_pca$pred), as.factor(Ytest))

table(as.factor(dfRidgeOpt_pca$pred), as.factor(Ytest))

pi_pca <- as.matrix(as.data.frame(pi_pca))
colnames(pi_pca) <- c('A', 'L', 'R', 'T', 'W')

auc_pca <- multiclass.roc(as.factor(Ytest), pi_pca, levels=c('A', 'L', 'R', 'T', 'W'))
rs_pca <- auc_pca[['rocs']]

plot.roc(rs_pca[[1]][[1]], levels=c('A', 'L', 'R', 'T', 'W'))
sapply(2:length(rs_pca),function(i) lines.roc(rs_pca[[i]][[1]],col=i))

cbind(Ytest,as.data.frame(pi_pca)) %>% roc_curve(Ytest, A:W) %>% autoplot()


####### Multinomial Model #################
Z <- pca_X$x[,1:350]

## Total number of available PCs
n_PC <- ncol(Z)

## cv.glm() requires the response and predictors in one data.frame, so we need
## to combine them back together
fit_data <- data.frame(Ytrain, Z)
head(fit_data)

## Example of PC Log. Reg. with all PCs
model <- multinom(Ytrain ~ ., data = fit_data, MaxNWts=84581)
# summary(full_model)

summary(model)
# zvalues <- summary(model)$coefficients / summary(model)$standard.errors
# Show z-values
# zvalues

# pnorm(abs(zvalues), lower.tail=FALSE)*2

fitted(model)

x_preds = as.data.frame(predict(model, pca_X_test$x[,1:350], 'probs'))

x_preds$preds = colnames(x_preds)[apply(x_preds,1,which.max)]
x_preds$actual = Ytest
table(x_preds$preds)
mean(x_preds$preds == Ytest)
table(x_preds$preds)
table(Ytest)
table(as.factor(x_preds$preds), as.factor(Ytest))

confusionMatrix(as.factor(x_preds$preds), as.factor(Ytest))




######## LASSO Model FULL ###################
library(doParallel)
registerDoParallel(4)
# cv_lasso <- cv.glmnet(
#   x = Xtrain,
#   y = Ytrain,
#   alpha = 1,               #alpha = 1 for LASSO
#   type.measure = "class", parallel = TRUE,
#   family = "multinomial", type.multinomial = "grouped")
# saveRDS(cv_lasso, 'models/SFinGe/cv_lasso_model.rds')
cv_lasso <- readRDS('models/SFinGe/cv_lasso_model.rds')
plot(cv_lasso)

# lasso <- glmnet(
#   x = Xtrain,
#   y = Ytrain,
#   lambda = cv_lasso$lambda.min,
#   alpha = 1, parallel = TRUE,       # alpha = 1 forLASSO
#   family="multinomial", type.multinomial = "grouped")
# saveRDS(lasso, 'models/SFinGe/lasso_model.rds')
lasso <- readRDS('models/SFinGe/lasso_model.rds')
lasso

dfLassoOpt <- data.frame(
  pred = predict(lasso,
                 newx = Xtest,
                 s = cv_lasso$lambda.min,
                 type = "class") %>% c(.),
  known.truth = Ytest)
pi = predict(lasso,
             newx = Xtest,
             s = cv_lasso$lambda.min,
             type = "response")
dfLassoOpt$pi <- pi

confusionMatrix(as.factor(dfLassoOpt$pred), Ytest)

table(as.factor(dfLassoOpt$pred), Ytest)

pi <- as.matrix(as.data.frame(pi))
colnames(pi) <- c('A', 'L', 'R', 'T', 'W')

auc <- multiclass.roc(as.factor(Ytest), pi, levels=c('A', 'L', 'R', 'T', 'W'))
rs <- auc[['rocs']]
plot.roc(rs[[1]][[1]], levels=c('A', 'L', 'R', 'T', 'W'))
sapply(2:length(rs),function(i) lines.roc(rs[[i]][[1]],col=i))
cbind(Ytest,as.data.frame(pi)) %>% roc_curve(Ytest, A:W) %>% autoplot()
summary(lasso)
coef(cv_lasso)
coef(lasso)

########## RIDGE MODEL FULL #################
# cv_ridge <- cv.glmnet(
#   x = Xtrain,
#   y = Ytrain,
#   alpha = 0,               # ridge: alpha = 0
#   type.measure = "class", parallel = TRUE,
#   family = "multinomial", type.multinomial = "grouped")
# saveRDS(cv_ridge, 'models/SFinGe/cv_ridge_model.rds')
cv_ridge <- readRDS('models/SFinGe/cv_ridge_model.rds')
plot(cv_ridge)

# ridge <- glmnet(
#   x = Xtrain,
#   y = Ytrain,
#   lambda = cv_ridge$lambda.min,
#   alpha = 0, parallel = TRUE,     # ridge: alpha = 0
#   family="multinomial", type.multinomial = "grouped")
# saveRDS(ridge, 'models/SFinGe/ridge_model.rds')
ridge <- readRDS('models/SFinGe/ridge_model.rds')

ridge

dfRidgeOpt <- data.frame(
  pred = predict(ridge,
                 newx = Xtest,
                 s = cv_ridge$lambda.min,
                 type = "class") %>% c(.),
  known.truth = Ytest)
pi = predict(ridge,
             newx = Xtest,
             s = cv_ridge$lambda.min,
             type = "response")
coef_ridge = predict(ridge,
             newx = Xtest,
             s = cv_ridge$lambda.min,
             type = "coefficients")
dfRidgeOpt$pi <- pi

confusionMatrix(as.factor(dfRidgeOpt$pred), as.factor(Ytest))

table(as.factor(dfRidgeOpt$pred), as.factor(Ytest))

pi <- as.matrix(as.data.frame(pi))
colnames(pi) <- c('A', 'L', 'R', 'T', 'W')

auc <- multiclass.roc(as.factor(Ytest), pi, levels=c('A', 'L', 'R', 'T', 'W'))
rs <- auc[['rocs']]
plot.roc(rs[[1]][[1]], levels=c('A', 'L', 'R', 'T', 'W'))
sapply(2:length(rs),function(i) lines.roc(rs[[i]][[1]],col=i))
cbind(Ytest,as.data.frame(pi)) %>% roc_curve(Ytest, A:W) %>% autoplot()

summary(ridge)
fimp <- as.data.frame(as.matrix((coef_ridge$A)))
fimp$name <- row.names(fimp)
colnames(fimp) <- c('coeff', 'name')
rownames(fimp) <- NULL
fimp <- fimp[order(-fimp$coeff),]


