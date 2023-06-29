import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    st.title('Prova sommativa 29/6')

    st.subheader('Plotting our dataset for viewing our data')
    df = pd.read_csv('Startup.csv')

    st.dataframe(df)

    X = df.drop(columns=['Profit'], axis=1)
    y = df['Profit']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=667
                                                        )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    length = y_pred.shape[0]
    x = np.linspace(0, length, length)

    res_df = pd.DataFrame(data=list(zip(y_pred, y_test)),
                          columns=['predicted', 'real'])
    st.subheader('Real values of dataset and predicted values of our model')
    st.dataframe(res_df)

    r2score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.subheader('Various type of errors for our model')
    st.write('R2_score: ', r2score)
    st.write('MAE(mean absolute): ', mae)
    st.write('MSE(mean squared): ', mse)
    st.write('RMSE: ', rmse)

    st.subheader('Plot of our Values(Real and Predicted)')
    fig = plt.figure(figsize=(10, 8))
    plt.plot(x, y_test, label='real y')
    plt.plot(x, y_pred, '-r', label="predicted y'")
    plt.legend(loc=2)
    st.pyplot(fig)

    # mlem.api.save(model,
    #               'model_',  # model_.mlem
    #               sample_data=X_train  # features
    #               )
    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    main()
