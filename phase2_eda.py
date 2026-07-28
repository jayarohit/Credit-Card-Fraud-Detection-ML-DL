import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("data/raw/creditcard.csv")


# displaying first 5 rows
print(df.head())


# displaying no of rows and columns 
print("Shape:",df.shape)


#displaying all columns
print("\nColumns:")
print(df.columns)


# displaying dataset info
print("\nDataset Information:")
df.info()


# checking for missing values
print("n\Missing Values:")
print(df.isnull().sum())


# counting duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())  # describing 
print("\nShape Before Removing:",df.shape)


# removing duplicate transactions
df = df.drop_duplicates()
print("Shape After Removing:",df.shape)


# no.of normal and fraudulent transactions
print("\nClass Distribution:")
print(df["Class"].value_counts())


# plotting class distribution  bar graph
class_counts = df["Class"].value_counts()
plt.bar(class_counts.index,class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.show()

# plotting distribution of transaction amounts histogram 
plt.figure(figsize=(8,5))
plt.hist(df["Amount"],bins = 50)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()


# plotting distribution of transaction time histogram
plt.figure(figsize=(8,5))
plt.hist(df["Time"],bins = 50)
plt.title("Distribution of Transaction Time")
plt.xlabel("Time in seconds")
plt.ylabel("Frequency")
plt.show()


# plotting transaction amounts by class using boxplot
a = df.boxplot(column = "Amount", by ="Class", figsize=(8,6))
plt.title("Transaction Amount by Class")
plt.suptitle("")
plt.xlabel("Class (0 = Normal,1=Fraud)")
plt.ylabel("Transaction Amount")
plt.show()


# plotting transaction amount by class(Log Scale)
b = df.boxplot(column = "Amount", by = "Class", figsize = (8,6))
plt.yscale("log")
plt.title("Transaction Amount by Class (Log Scale)")
plt.suptitle("")
plt.xlabel("Class")
plt.ylabel("Transaction Amount")
plt.show()


# correlation analysis using heatmap
plt.figure(figsize = (14,10))
correlation = df.corr()
sns.heatmap(correlation, cmap = "coolwarm", )
plt.title("Correlation Heatmap")
plt.show()