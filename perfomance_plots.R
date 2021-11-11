### Part one (CASTOR Algorithm): Performance plots
# Load required libraries
library("ggplot2")
setwd("C:/Users/Albert Doughan/Desktop/ALL/SOP/Yemaachi")

# Import the dataset
df  = read.csv(file = "performance.csv")
df

# Generate the plot
ggplot(data = df, aes(x= ML.Algorithm, y = FPR)) + 
  geom_bar(stat="identity", fill ="#0073C2FF") + theme_bw()+
  theme(
    axis.title.x = element_text(size=25),
    axis.title.y = element_text(size=25),
    axis.text=element_text(size=15)) +
  xlab("ML Algorithm")+ ylab("F-score")


### Part one (Naive bayes and logistic regression): Performance plots
# Import the dataset
df1  = read.csv(file = "C:/Users/Albert Doughan/Desktop/ALL/SOP/Yemaachi/16S sequnces/16S performance.csv")
df1

# Generate the plot
ggplot(data = df1, aes(x= Algorithm , y = F1.score)) + 
  geom_bar(stat="identity", fill ="#0073C2FF") + theme_bw()+
  theme(
    axis.title.x = element_text(size=25),
    axis.title.y = element_text(size=25),
    axis.text=element_text(size=15)) +
  xlab("Algorithm")+ ylab("F1-score")



