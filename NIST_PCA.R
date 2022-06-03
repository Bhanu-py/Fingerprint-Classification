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

df1 <- read.csv('data/NISTDB4-F.csv', header=FALSE)
df1$quality <- 'F'

df2 <- read.csv('data/NISTDB4-S.csv', header=FALSE)
df2$quality <- 'S'

dim(df1)

tot <- rbind(df1,df2)
colnames(tot)[ncol(tot)-1] <- 'label'
labels <- tot$label
tot$quality <- as.factor(tot$quality)
tot$label <- as.factor(tot$label)
table(tot$quality, tot$label)
table(tot$label)

dim(tot)
sum(is.na(tot))
colSums(is.na(tot))

map_chr(tot, typeof) %>% 
  tibble() %>% 
  table()

#### Missing Values Treatment #####
df <- modify(tot[,1:(ncol(tot)-2)], is.na) %>% 
  colSums() %>%
  tibble(names = colnames(tot[,1:(ncol(tot)-2)]),missing_values=.) %>% 
  arrange(-missing_values)

hist(df[df$missing_values != 0,]$missing_values,labels = TRUE, ylim = c(0,80), xlim = c(0,3300), breaks = 50,
     main='Number of missing values in each column', xlab = 'columns')

# Removing columns with 50% of total rows having missing values
names <- modify(tot[,1:(ncol(tot)-2)], is.na) %>% 
  colSums() %>%
  tibble(names = colnames(tot[,1:(ncol(tot)-2)]), missing_values=.) %>% 
  filter(missing_values < 1500) %>% 
  select(1)

tot <- tot[c(names$names, 'quality', 'label')]


## Imputing remaining predictors with less than 1500 missing values
library(naniar)
vis_miss(tot[,846:1019])
tot <- tot %>% 
      mutate_if(is.numeric, function(x) ifelse(is.na(x), median(x, na.rm = T), x))
sum(is.na(tot))

# # #### Remove Outliers ######
cap <- function(x){
quantiles <- quantile( x, c(.05, 0.25, 0.75, .95 ))
x[ x < quantiles[2] - 1.5*IQR(x) ] <- quantiles[1]
x[ x > quantiles[3] + 1.5*IQR(x) ] <- quantiles[4]
x}

tot <- tot %>% mutate_if(is.numeric, cap)

boxplot(tot[, 1:6])

##### ZERO VARIANCE #####
##### Finding if columns have Zero Variance that gives NAs while scaling
x <- cbind(lapply(tot[,1:(ncol(tot)-2)], FUN = var, na.rm = T))

vardf <- data.frame('col' = rownames(x), 'variation' = unlist(x))
vardf$col[round(vardf$variation, 5) == 0.0000]
zero_var <- vardf[order(vardf$variation),]
# str(tot$V1312)

# remove columns with zero variance
quality <- tot$quality
label <- tot$label
tot  <- tot[,!(round(vardf$variation, 5) == 0.0000)]
tot$label <- label
tot$quality <- quality
rm(quality)
rm(label)
dim(tot) 

##### CLASS BALANCING #####
table(tot$label)
prop.table(table(tot$label))
tot_f <- tot[tot$quality == 'F',]
tot_s <- tot[tot$quality == 'S',]
dim(tot_f)
smoted_fT <- oversample_smote((tot_f %>% select(-'quality')), "T", "label", 150)
smoted_sT <- oversample_smote((tot_s %>% select(-'quality')), "T", "label", 150)

smoted_fT$quality = 'F'
smoted_sT$quality = 'S'
table(smoted_fT$label)
table(smoted_sT$label)
tot <- rbind(tot, smoted_fT, smoted_sT)

table(tot$label)
prop.table(table(tot$label))

# tot %>%
#   glimpse()

## Glimpse first 10 columns
head(tot[, 1:10], n = 14)

saveRDS(tot, 'processed_NIST.rds')
tot <- readRDS('processed_NIST.rds')

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
length(Ytest)

table(Ytrain)
table(Ytest)


Xtrain <- scale(Xtrain, center = TRUE, scale = TRUE)
Xtest <- scale(Xtest, center = TRUE, scale = TRUE)
sum(is.na(Xtrain))
sum(is.na(Xtest))


#### Scaling X for PCA #############
# X <- scale(X, center = TRUE, scale = TRUE)

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

grid.arrange(pVar2, pVar3, nrow=1, top = grid::textGrob("NIST",gp=grid::gpar(fontsize=20,font=3)))

# ######################## PC plots ########################################################
k <- 400
Vk <- svdX$v[,1:k]
Uk <- svdX$u[,1:k]
Dk <- diag(svdX$d[1:k])
Zk <- Uk%*%Dk
colnames(Zk) <- paste0("PC",1:k)
tit <- 'PC plots'

Zk %>%
  as.data.frame %>%
  mutate(Label = Ytrain %>% as.factor) %>%
  ggplot(aes(x= PC1, y = PC2, color = Label)) +
  geom_point(size = 3)

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
    title= tit,
    hovermode='closest',
    dragmode= 'select',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=4),
    yaxis=list(domain=NULL, showline=F, zeroline=F, gridcolor='#ffff', ticklen=4),
    xaxis2=axis,
    xaxis3=axis,
    xaxis4=axis,
    yaxis2=axis,
    yaxis3=axis,
    yaxis4=axis
  )
options(warn=-1)
fig
# 
# ####################### 3D plot ######################################################

fig <- plot_ly(Zk %>%
                 as.data.frame %>%
                 mutate(Label = Ytrain %>% as.factor), x = ~PC1, y = ~PC3, z = ~PC5, color = ~Ytrain, colors = c('#636EFA','#EF553B','#00CC96') ) %>%
  add_markers(size = 12)


fig <- fig %>%
  layout(
    title = '3-D plot on PC 1,2,3',
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

cv_LASSO_pca <- cv.glmnet(
  x = Z_train,
  y = Ytrain,
  alpha = 1,               # Lasso: alpha = 1
  type.measure = "class",
  family = "multinomial", type.multinomial = "grouped")
saveRDS(cv_LASSO_pca, 'models/NIST/cv_lasso_pca_model.rds')
cv_lasso_pca <- readRDS('models/NIST/cv_lasso_pca_model.rds')
plot(cv_lasso_pca)

lasso_pca <- glmnet(
  x = Z_train,
  y = Ytrain,
  lambda = cv_LASSO_pca$lambda.min,
  alpha = 1,         #alpha = 1 forLASSO
  family="multinomial", type.multinomial = "grouped")
saveRDS(lasso_pca, 'models/NIST/lasso_pca_model.rds')
lasso_pca <- readRDS('models/NIST/lasso_pca_model.rds')

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
## Cross validation 
cv_ridge_pca <- cv.glmnet(
  x = Z_train,
  y = Ytrain,
  alpha = 0,               # ridge: alpha = 0
  type.measure = "class",
  family = "multinomial", type.multinomial = "grouped")
saveRDS(cv_ridge_pca, 'models/NIST/ridge_PCA_model.rds')
cv_ridge_pca <- readRDS('models/NIST/ridge_PCA_model.rds')

plot(cv_ridge_pca)
#Fitting with best lambda values
ridge_pca <- glmnet(
  x = Z_train,
  y = Ytrain,
  lambda = cv_ridge_pca$lambda.min,
  alpha = 0,         #alpha = 0 for ridge
  family="multinomial", type.multinomial = "grouped")
saveRDS(ridge_pca, 'models/NIST/ridge_PCA.rds')
ridge_pca <- readRDS('models/NIST/ridge_PCA.rds')

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
cv_lasso <- cv.glmnet(
  x = Xtrain,
  y = Ytrain,
  alpha = 1,               #alpha = 1 for LASSO
  type.measure = "class", parallel = TRUE,
  family = "multinomial", type.multinomial = "grouped")
saveRDS(cv_lasso, 'models/NIST/cv_lasso_model.rds')
cv_lasso <- readRDS('models/NIST/cv_lasso_NIST.rds')
plot(cv_lasso)

lasso <- glmnet(
  x = Xtrain,
  y = Ytrain,
  lambda = cv_lasso$lambda.min,
  alpha = 1, parallel = TRUE,       # alpha = 1 forLASSO
  family="multinomial", type.multinomial = "grouped")
saveRDS(lasso, 'models/NIST/lasso_model.rds')
lasso <- readRDS('models/NIST/lasso_NIST.rds')
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
cv_ridge <- cv.glmnet(
  x = Xtrain,
  y = Ytrain,
  alpha = 0,               # ridge: alpha = 0
  type.measure = "class", parallel = TRUE,
  family = "multinomial", type.multinomial = "grouped")
saveRDS(cv_ridge, 'models/NIST/cv_ridge_NIST.rds')
cv_ridge <- readRDS('models/NIST/cv_ridge_NIST.rds')
plot(cv_ridge)

ridge <- glmnet(
  x = Xtrain,
  y = Ytrain,
  lambda = cv_ridge$lambda.min,
  alpha = 0, parallel = TRUE,     # ridge: alpha = 0
  family="multinomial", type.multinomial = "grouped")
saveRDS(ridge, 'models/NIST/ridge_NIST.rds')
ridge <- readRDS('models/NIST/ridge_NIST.rds')

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


