cat("\n")
cat("============================================\n")
cat("          CHEENI KUM 2.0\n")
cat(" Data bolega, cheeni ka scene kholega!\n")
cat("============================================\n")
cat("Diabetes Prediction Using Logistic Regression\n\n")


# Diabetes Prediction Using Logistic Regression in R

packages <- c("ggplot2","dplyr","caret","pROC")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg, dependencies=TRUE)
  library(pkg, character.only=TRUE)
}
diabetes <- read.csv(
  "C:/Users/hp/Documents/diabetes_probabilistic.csv",
  na.strings = c("", "NA")
)
names(diabetes) <- trimws(names(diabetes))
required <- c("Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
              "BMI","DiabetesPedigreeFunction","Age","Outcome")
if (!all(required %in% names(diabetes))) stop("Select diabetes_updated.csv.")
diabetes <- diabetes[!is.na(diabetes$Outcome),]
if (!all(diabetes$Outcome %in% c(0,1))) stop("Outcome must be 0 or 1.")
diabetes$Outcome <- as.integer(diabetes$Outcome)
cat("Dimensions:",dim(diabetes),"\nOutcome counts:\n")
print(table(diabetes$Outcome))
cat("Missing values:\n"); print(colSums(is.na(diabetes)))

print(ggplot(diabetes,aes(factor(Outcome)))+geom_bar()+
 labs(title="Diabetes Outcome Distribution",x="Outcome (0/1)",y="Records")+theme_minimal())
print(ggplot(diabetes,aes(factor(Outcome),Glucose))+geom_boxplot(na.rm=TRUE)+
 labs(title="Glucose Levels by Outcome",x="Outcome",y="Glucose")+theme_minimal())
print(ggplot(diabetes,aes(BMI,fill=factor(Outcome)))+
 geom_histogram(bins=25,alpha=.7,na.rm=TRUE,position="identity")+
 labs(title="BMI Distribution by Outcome",x="BMI",y="Count",fill="Outcome")+theme_minimal())

num <- diabetes[,vapply(diabetes,is.numeric,logical(1)),drop=FALSE]
cor_df <- as.data.frame(as.table(cor(num,use="pairwise.complete.obs")))
names(cor_df) <- c("Variable1","Variable2","Correlation")
print(ggplot(cor_df,aes(Variable1,Variable2,fill=Correlation))+geom_tile()+
 scale_fill_gradient2(low="blue",mid="white",high="red",midpoint=0)+
 labs(title="Correlation Heatmap",x=NULL,y=NULL)+theme_minimal()+
 theme(axis.text.x=element_text(angle=45,hjust=1)))

set.seed(123)
idx <- caret::createDataPartition(factor(diabetes$Outcome,levels=c(0,1)),p=.8,list=FALSE)
train_data <- diabetes[idx,,drop=FALSE]
test_data <- diabetes[-idx,,drop=FALSE]
predictors <- required[1:8]
for (col in predictors) {
 med <- median(train_data[[col]],na.rm=TRUE)
 if (!is.finite(med)) stop(paste("Cannot calculate median for",col))
 train_data[[col]][is.na(train_data[[col]])] <- med
 test_data[[col]][is.na(test_data[[col]])] <- med
}
model <- glm(Outcome~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+
 BMI+DiabetesPedigreeFunction+Age,data=train_data,family=binomial(link="logit"))
cat("\nMODEL SUMMARY\n"); print(summary(model))
probabilities <- predict(model,newdata=test_data,type="response")

# ==========================================
# THRESHOLD COMPARISON EXPERIMENT
# ==========================================

thresholds <- c(0.20, 0.30, 0.40, 0.50)

actual <- factor(test_data$Outcome, levels = c(0, 1))

for (t in thresholds) {
  
  cat("\nThreshold:", t, "\n")
  
  pred <- factor(
    ifelse(probabilities >= t, 1, 0),
    levels = c(0, 1)
  )
  
  cm_test <- caret::confusionMatrix(
    data = pred,
    reference = actual,
    positive = "1"
  )
  
  cat("Accuracy:",
      round(cm_test$overall["Accuracy"], 4), "\n")
  
  cat("Sensitivity:",
      round(cm_test$byClass["Sensitivity"], 4), "\n")
  
  cat("Specificity:",
      round(cm_test$byClass["Specificity"], 4), "\n")
}

roc_obj <- pROC::roc(test_data$Outcome,probabilities,levels=c(0,1),direction="<",quiet=TRUE)
cat("ROC-AUC:",as.numeric(pROC::auc(roc_obj)),"\n")
plot(roc_obj,main="ROC Curve - Diabetes Prediction",col="blue",lwd=2)

sample_record <- data.frame(Pregnancies=2,Glucose=120,BloodPressure=70,
 SkinThickness=25,Insulin=90,BMI=28.5,DiabetesPedigreeFunction=.35,Age=30)
sample_prob <- predict(model,newdata=sample_record,type="response")
cat("Sample model probability:",round(sample_prob,4),"\n")
cat("Sample class:",ifelse(sample_prob>=threshold,1,0),"\n")
write.csv(data.frame(Actual=test_data$Outcome,Predicted=predicted_class,
 Probability=round(probabilities,4)),"diabetes_predictions.csv",row.names=FALSE)
saveRDS(model,"diabetes_logistic_model.rds")
cat("Project completed. Synthetic educational data only; not a medical diagnosis.\n")


# FINAL MODEL EVALUATION AT THRESHOLD 0.30

final_threshold <- 0.30

final_predictions <- ifelse(
  probabilities >= final_threshold, 1, 0
)

actual <- factor(test_data$Outcome, levels = c(0, 1))
predicted <- factor(final_predictions, levels = c(0, 1))

final_cm <- caret::confusionMatrix(
  data = predicted,
  reference = actual,
  positive = "1"
)

cat("\nFINAL CONFUSION MATRIX\n")
print(final_cm)

cat("\nFINAL RESULTS\n")
cat("Threshold:", final_threshold, "\n")
cat("Accuracy:", round(final_cm$overall["Accuracy"], 4), "\n")
cat("Sensitivity:", round(final_cm$byClass["Sensitivity"], 4), "\n")
cat("Specificity:", round(final_cm$byClass["Specificity"], 4), "\n")

# Save the final test predictions
write.csv(
  data.frame(
    Actual = test_data$Outcome,
    Predicted = final_predictions,
    Probability = probabilities
  ),
  "diabetes_final_predictions.csv",
  row.names = FALSE
)

