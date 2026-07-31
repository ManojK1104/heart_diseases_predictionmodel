from src.data_ingestion import ingestion
from src.data_preprocessing import processing
from src.model_building import model_build

def main():

    df=ingestion()
    print(df.shape)

    X_train,X_test,y_train,y_test=processing(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    model,accuracy=model_build(X_train,X_test,y_train,y_test)
    return model,accuracy
main()