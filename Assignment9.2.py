try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets
    from sklearn.metrics import mean_squared_error
    
    boston = datasets.load_boston()
    features = pd.DataFrame(boston.data, columns=boston.feature_names)
    targets = boston.target

    
    #Spliiting training and test data
    X_train,X_test,y_train,y_test= train_test_split(features,targets,test_size= .20,random_state=0)
    
    #Scaling
    sc= StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    
    #Fitting RF clasiification to the trainaing set
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=10,criterion='mse',random_state=0)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    
    #Mean Squared error
     print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
    
except Exception as e:
    print(e)
