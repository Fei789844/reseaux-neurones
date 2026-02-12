bc <- read.csv("~/Downloads/神经网络Projet/Housing.csv", stringsAsFactors = FALSE, check.names = TRUE)
pred_cols <- setdiff(names(bc), "price")
char_cols <- pred_cols[sapply(bc[, pred_cols], is.character)]
bc[char_cols] <- lapply(bc[char_cols], factor)

X0 <- model.matrix(price ~ ., data = bc)[, -1] 
y  <- bc$price
dim(X0)  
colnames(X0)  
head(X0)

set.seed(123)
train_idx <- sample(1:nrow(bc), floor(0.7 * nrow(bc)))
y_train <- y[train_idx]
y_test  <- y[-train_idx]
X_train0 <- X0[train_idx, , drop = FALSE]
X_test0  <- X0[-train_idx, , drop = FALSE]
X_train <- scale(X_train0)
X_test  <- scale(X_test0,
                 center = attr(X_train, "scaled:center"),
                 scale  = attr(X_train, "scaled:scale"))


library(keras3)
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.2) %>%  
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mse', 
  optimizer = optimizer_adam(learning_rate = 0.1),
  metrics = c('mae')
)

history <- model %>% fit(
  X_train, y_train,
  epochs = 100, 
  batch_size = 32, 
  validation_data = list(X_test, y_test),
  verbose = 1, 
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10))
)

predictions <- model %>% predict(X_test)
predictions_df <- data.frame(
  Actual_Price = y_test,  
  Predicted_Price = predictions  
)
print(predictions_df)



rss <- sum((predictions_df$Actual_Price - predictions_df$Predicted_Price)^2)
tss <- sum((predictions_df$Actual_Price - mean(predictions_df$Actual_Price))^2)
r_squared <- 1 - (rss / tss)
print(paste("R-squared: ", r_squared))

plot(predictions_df$Actual_Price, predictions_df$Predicted_Price,
     xlab = "Actual Price", ylab = "Predicted Price",
     main = "Actual vs Predicted Prices", pch = 19, col = rgb(0, 0, 1, 0.5))
abline(0, 1, col = "red")  


