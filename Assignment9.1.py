try:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from sklearn.tree import DecisionTreeClassifier
    
    url='https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
    titanic = pd.read_csv(url,usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare','Survived'])
    
    #Separate out the depenent and independent features
    X=titanic.iloc[:,1:].values
    y=titanic.iloc[:,0].values
    
    #Taking Care of Missing data
    imputer= Imputer()
    
    #fit and transform in a single line
    X[:,[2]]=imputer.fit_transform(X[:,[2]])
    X[:,2]=X[:,2].astype(int)
    
    #Encoding the categorial data
    labelencoder_X=LabelEncoder()
    X[:,1]=labelencoder_X.fit_transform(X[:,1])
    
    #Split data into train and test
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size= .20,random_state=0)
    
    #Performing standard scaling
    sc=StandardScaler()
    X_train =sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Create Decision tree model
    classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    
    
    #Accuracy and confusion matrix
    cm= confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    class_report= classification_report(y_test,y_pred)
    
except Exceptionon as e:
    print(e)
