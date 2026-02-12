library(DALEX)
library(randomForest)
library(tidyverse)
df <- read.csv("~/Downloads/神经网络Projet/Used_Car_Price_Prediction.csv", stringsAsFactors = TRUE)
data_clean <- df %>%
  select(-car_name, -variant, -model, -rto, -registered_city, -ad_created_on, 
         -is_hot, -broker_quote, -emi_starts_from, -booking_down_pymnt, -total_owners) %>% 
  mutate(
    sale_price = as.numeric(sale_price),
    yr_mfr = as.numeric(yr_mfr),
    kms_run = as.numeric(kms_run),
    original_price = as.numeric(original_price),
    sale_price = as.numeric(sale_price)
  ) %>%

  filter(body_type != "") %>%
  filter(fuel_type != "") %>%
  filter(transmission != "") %>%
  na.omit()
cat("le nombre de caractéristiques :", ncol(data_clean) - 1)
cat("la liste des caractéristiques :", paste(colnames(data_clean)))

set.seed(123)
model <- randomForest(sale_price ~ ., data = data_clean, ntree = 100)
nrow(data_clean)
ncol(data_clean)
#XAI
explainer <- DALEX::explain(
  model = model,
  data = data_clean %>% select(-sale_price), 
  y = data_clean$sale_price,                 
  label = "le modèle de véhicules d'occasion"
)

importance <- model_parts(explainer, B = 5) 
plot(importance) + 
  ggtitle("les variables clés") 
  

car1 <- data_clean[3, ]
print(car1)

num_vars <- names(data_clean %>% select(where(is.numeric), -sale_price))
cp <- predict_profile(explainer, 
                      new_observation = car1, 
                      variables = num_vars) 
plot(cp) + 
  ggtitle("Analyse de sensibilité : Tous les paramètres numériques") + 
  labs(y = "Prix prédit", x = "Valeur de la variable") +
  theme_minimal()


cat_vars <- c("fuel_type", "transmission", "body_type")
valid_vars <- intersect(cat_vars, colnames(data_clean))
cp_cat <- predict_profile(explainer, 
                          new_observation = car1, 
                          variables = valid_vars)
library(ggplot2)
plot_data <- as.data.frame(cp_cat) %>%
  pivot_longer(cols = all_of(valid_vars), values_to = "Category_Value") %>%
  filter(name == `_vname_`) 
ggplot(plot_data, aes(x = Category_Value, y = `_yhat_`, fill = Category_Value)) +
  geom_col(width = 0.6) + 
  facet_wrap(~ `_vname_`, scales = "free_x") +
  geom_hline(yintercept = predict(explainer, car1), 
             linetype = "dashed", color = "red", alpha = 0.5) +
  ggtitle("Impact des caractéristiques catégorielles", ) + 
  labs(y = "Prix prédit", x = "Catégorie") +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.text.x = element_text(angle = 45, hjust = 1))


