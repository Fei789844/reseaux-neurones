bc <- read.csv("~/Downloads/神经网络Projet/Thyroid_Diff.csv",
               stringsAsFactors = FALSE, check.names = TRUE)
bc$Recurred <- factor(bc$Recurred,
                      levels = c("No", "Yes"),
                      labels = c("Non.Recurred", "Recurred"))
pred_cols <- setdiff(names(bc), "Recurred")
char_cols <- pred_cols[sapply(bc[, pred_cols], is.character)]
bc[char_cols] <- lapply(bc[char_cols], factor)

X0 <- model.matrix(Recurred ~ . , data = bc)[, -1]
y  <- bc$Recurred
ncol(bc) - 1
ncol(X0)
set.seed(123)
idx_1 <- which(y == "Recurred")
idx_0 <- which(y == "Non.Recurred")

train_idx <- c(
  sample(idx_1, floor(0.7 * length(idx_1))),
  sample(idx_0, floor(0.7 * length(idx_0)))
)

y_train <- y[train_idx]
y_test  <- y[-train_idx]

X_train0 <- X0[train_idx, , drop = FALSE]
X_test0  <- X0[-train_idx, , drop = FALSE]
X_train <- scale(X_train0)
X_test  <- scale(X_test0,
                 center = attr(X_train, "scaled:center"),
                 scale  = attr(X_train, "scaled:scale"))

pca <- prcomp(X_train, center = FALSE, scale. = FALSE)
pca_all <- rbind(
  data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2],
             Set = "Train", true = y_train),
  data.frame(PC1 = predict(pca, X_test)[, 1],
             PC2 = predict(pca, X_test)[, 2],
             Set = "Test", true = y_test))

#PMC
library(nnet)
y_train_ind <- class.ind(y_train)
colnames(y_train_ind) <- levels(y_train)

pmc <- nnet(x = X_train, y = y_train_ind,
            size = 8, decay = 0.01, maxit = 1000,
            softmax = TRUE, trace = TRUE)
p_mlp <- predict(pmc, X_test, type = "raw")[, "Recurred"]
pred_mlp <- factor(ifelse(p_mlp >= 0.5, "Recurred", "Non.Recurred"),
                   levels = levels(y_test))

#LDA
library(MASS)
lda_fit <- lda(x = X_train, grouping = y_train)
lda_out <- predict(lda_fit, X_test)

p_lda <- lda_out$posterior[, "Recurred"]
pred_lda <- factor(lda_out$class, levels = levels(y_test))

# ROC
library(pROC)
roc_mlp <- roc(y_test, p_mlp, levels = c("Non.Recurred","Recurred"), direction = "<")
roc_lda <- roc(y_test, p_lda, levels = c("Non.Recurred","Recurred"), direction = "<")
plot(roc_mlp, col = "blue", main = "ROC Curve Comparison")
plot(roc_lda, col = "red", add = TRUE)
legend("bottomright", legend = c("MLP", "LDA"), col = c("blue", "red"), lwd = 2)

test_pca <- subset(pca_all, Set == "Test")
test_pca$pred_mlp <- pred_mlp
test_pca$pred_lda <- pred_lda
ggplot(test_pca, aes(PC1, PC2, color = pred_mlp, shape = true)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "ACP (test data): PMC vs Vrai ",
       x = "PC1", y = "PC2",
       color = "Prévision (PMC)", shape = "Vrai") +
  theme_minimal()

ggplot(test_pca, aes(PC1, PC2, color = pred_lda, shape = true)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "ACP (test data): LDA vs Vrai ",
       x = "PC1", y = "PC2",
       color = "Prévision (LDA)", shape = "Vrai") +
  theme_minimal()


confusion_mlp <- table(True = y_test, Pred = pred_mlp)
print(confusion_mlp)
confusion_lda <- table(True = y_test, Pred = pred_lda)
print(confusion_lda)

