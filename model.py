# loading libraries...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# reading dataset from csv
df = pd.read_csv("./dataset.csv")

# encoding labels to numbers
lr = LabelEncoder()
new_labels = lr.fit_transform(df["Label"])

df["New_Labels"] = new_labels

X = df.drop(['New_Labels','Label'],axis="columns")
y = df["New_Labels"]

# scaling the features
sc = StandardScaler()
X_transform = sc.fit_transform(X)

# dividing the dataset into Testing and Training
X_train,X_test,y_train,y_test = train_test_split(X_transform,y,test_size=0.2)

# making object of Support Vector Classifier
model = SVC()
model.fit(X_train,y_train)

# Getting Accuracy
print(model.score(X_test,y_test))

# making prediction
p = model.predict(X_test)

# making confusion matrix
cm = confusion_matrix(y_test,p)

sns.heatmap(cm,cmap="Blues",annot=True)
plt.show()


