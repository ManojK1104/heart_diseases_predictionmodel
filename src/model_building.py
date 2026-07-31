from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


def model_build(X_train,X_test,y_train,y_test):
    model=RandomForestClassifier(random_state=1)

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    print("accuracy:",accuracy)
    print("report\n",report)

    os.makedirs("model",exist_ok=True)
    with open("model/model.pkl","wb")as f:
        pickle.dump(model,f)

    return model,accuracy