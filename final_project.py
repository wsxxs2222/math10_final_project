# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 18:04:16 2021

@author: 10723
"""

import streamlit as st
import pandas as pd
#import tensorflow as tf
#import sklearn as sk
import altair as alt
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
# title
st.title("prediction of league of legends match result using machine learning")
st.header("A short introduction to league of legends and what the dataset I'm are using is about")
st.write("league of legends is a game where each team picks 5 champions and battles in a \
         field with 3 lanes and 12 jungle monster camps. The team that destroys the enemy \
         base wins the game. The laning phase of the game is usually the first 8-12 minutes \
         which is the time this data set focused on. The laning phase is crucial for the team \
         since a early lead can be snowballed into a more substantial lead. The key events that \
         can happen in the laning phase are killing enemy champions, farming minons to gain gold \
         killing dragons/herald etc. those data are all included in the dataset")
st.header("This is the dataset I'm are using")
# import file and clean data
st.write('after loading the data set we first find the columns that are numerical and only take such columns')
df = pd.read_csv(r"C:\Users\10723\Downloads\high_diamond_ranked_10min.csv")
#df = pd.read_csv(r"C:\Users\joe fang\Downloads\high_diamond_ranked_10min.csv")
num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
df = df.loc[:,num_cols]
st.write(df)

# graphs about the basic facts of this dataset
# show gold lead is the most important factor
st.header("1. explore the dataset using charts")
prob_list = []
for i in range(-5000,5000,1000):
    x=df[(i<df['redGoldDiff']) & (df['redGoldDiff']<i+1000)]
    num_of_games=x.shape[0]
    num_of_wins=num_of_games-sum(x['blueWins'])
    prob=num_of_wins/num_of_games
    prob_list.append(prob)
x_index = [f'{i}~{(i+1)}k' for i in range(-5,5,1)]
x_pos = [i for i, _ in enumerate(x_index)]
fig, ax = plt.subplots()
plt.bar(x_pos, prob_list, color='green')
plt.xlabel("gold difference")
plt.ylabel("probability of winning the game")
plt.title("relationship between gold difference in 10 minutes and winning rate")
plt.xticks(x_pos, x_index)
plt.figure(figsize=(28,8))
st.pyplot(fig)
st.write("gold difference certainly has a positive correlation with winning chance")

chart1 = alt.Chart(df).mark_circle(size=80).encode(
    x='blueKills',
    y='blueDeaths',
    color='blueWins',
    tooltip=["blueFirstBlood", "redGoldDiff", "redExperienceDiff"]
    )
st.altair_chart(chart1)
st.write("generally, the more kills you have and the less deaths you have you are more likely to win")
st.write("however, the more kills/less deaths will affect the gold difference \
         those variables are dependent")
df['red_diff_kill_death']=df["blueDeaths"]-df["blueKills"]
chart3 = alt.Chart(df).mark_area().encode(
    x="redGoldDiff",
    y='red_diff_kill_death'
)
st.altair_chart(chart3)
st.write("this graph tells us gold difference and kill/death difference indeed go hand in hand. \
         based on the common sense of the game knowing either of these information will have the similar effect")       
chart2 = alt.Chart(df).mark_circle(size=80).encode(
    x='blueWardsPlaced',
    y='blueWardsDestroyed',
    color='blueWins'
    )
st.altair_chart(chart2)
st.write("the relation between ward placed/destroyed and blue team winning is not so clear \
         , even if you placed/destroyed a lot of ward, the result of the game still can go either directions")         
# new column with red kills minus red death


# the relationship between other factors and the chance of winning
# divide the data into test and training set
df1 = df.drop(['gameId', 'blueWins'], axis=1)
y_col = df["blueWins"]
st.write("")
st.header("2. The no brainer prediction")
st.write("one the most important factor that determines the outcome of the game is gold difference between two teams. Therefore, the no brainer prediction would be always choose the team that has a gold lead as the winning team. The goal of the project is to figure out a way to beat this no brainer prediction and see how much better can we predict the game.")
st.write("first see how good the no brainer prediction is")
red_lead = df[df['redGoldDiff']>0]
blue_lead = df[df['redGoldDiff']<0]

blue_wins = blue_lead['blueWins'].sum()
red_games = red_lead.shape[0]
blue_games = blue_lead.shape[0]
red_wins = red_games-red_lead['blueWins'].sum()
percentage = (red_wins+blue_wins)/(blue_games+red_games)
st.markdown(f"after some calculation, the no brainer prediction gives an accuracy of **{percentage}**")
st.write("")
st.header("3. predict the match results using machine learning")
st.write("now we try to use some machine learning methods to predict the match results using part of the dataset as training data")
st.write("we ask the user to choose how the testing/training data is selected \
         since we are using train_test_split")
t_size=st.slider("please select the fraction of data we use\
                    for the testing data", 
                    0.25,1.00,step=0.01
                    
                    
                    )
r_percent=st.slider("please select how random you want the data that will be used for test cases", 
                    40,100
                    
                    
                    )
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df1, y_col, test_size=t_size, random_state=r_percent)
# scale the data
scaler = StandardScaler()
k_train = x_train.loc[:,['blueDragons', 'blueHeralds', 'redGoldDiff', 'redExperienceDiff']]
k_test = x_test.loc[:,['blueDragons', 'blueHeralds', 'redGoldDiff', 'redExperienceDiff']]
scaler.fit(k_test)
k_test1 = scaler.transform(k_test)
scaler.fit(k_train)
k_train1 = scaler.transform(k_train)
st.write("we let the users to choose which method they want to use")
options = ["kmeans", "logistic regression", "neural network"]
method_selected = st.selectbox("please choose the method you wish to apply", options)
if method_selected == "kmeans":
    st.header("kmeans")
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(2)
    kmeans.fit(k_train1)
    k_result = kmeans.predict(k_test1)
    k_accuracy = (k_result == y_test).sum()/len(k_result)
    st.write("we use the training data and try to classify all the points into two clusters \
             , one cluster representing blue team win and the other represent blue team lose")
    st.markdown(f"after computations, the prediction accuracy using kmeans is **{k_accuracy}**")
elif method_selected == "logistic regression":
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(k_train1, y_train)
    l_result = clf.predict(k_test1)
    l_accuracy = (l_result == y_test).sum()/len(l_result)
    st.markdown(f"we performed the logistic regression using the training set, the accuracy of the prediction when we use the test set is **{l_accuracy}**")
elif method_selected == "neural network":
    neuron1 = st.slider("how many neurons do you want to have in hidden layer 1?",16,24)
    neuron2 = st.slider("how many neurons do you want to have in hidden layer 2?",16,24)
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape = (4,)),
            keras.layers.Dense(neuron1, activation="sigmoid"),
            keras.layers.Dense(neuron2, activation="sigmoid"),
            keras.layers.Dense(2,activation="softmax")
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=keras.optimizers.SGD(
                    learning_rate=0.01
                ),
              metrics=["accuracy"])
    model.fit(k_train1,y_train,epochs=10)
    n_accuracy = model.evaluate(k_train1, y_train)[1]
    st.markdown(f"we performed neural network with 2 hidden layers with {neuron1} and {neuron2} neurons and \
                obtained an accuracy of {n_accuracy}")

st.header("4. Is the neural network overfitting?")
model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape = (4,)),
            keras.layers.Dense(16, activation="sigmoid"),
            keras.layers.Dense(16, activation="sigmoid"),
            keras.layers.Dense(2,activation="softmax")
        ]
)
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=keras.optimizers.SGD(
                    learning_rate=0.01
                ),
              metrics=["accuracy"])
model.fit(k_train1,y_train,epochs=10)
history = model.fit(k_train1,y_train,epochs=100, validation_split = 0.2)
fig1, ax = plt.subplots()
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.pyplot(fig1)
st.write("according to the result I got, the validation set performs similarly with the training set, so we are not\
         so concerned with overfitting")
st.header("5. Conclusion")
st.write("unfortunately, none of the machine learning methods performs \
         significantly better than the no brainer method, in my opinion \
         , this is because the most important information we get from \
         the dataset is gold different between teams and \
         other information is either a cause of the gold difference \
         it is rather a minor factor (like which team gets the first dragon)")
st.write("if we want to improve the result, I think we should gather information about each team's composition \
         (what heroes did they pick), since this is another important piece of information \
         and it is independent of gold difference")
st.header("Reference")
my_expander = st.expander("expand")
with my_expander:
    st.write("link to the dataset: https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min/version/1")
    st.write("epander: https://blog.streamlit.io/introducing-new-layout-options-for-streamlit/")
    st.write("logistic regression code: https://christopherdavisuci.github.io/UCI-Math-10/Week8/logistic.html")
    st.write("neural network code: https://christopherdavisuci.github.io/UCI-Math-10/Week9/Week8-Friday.html")
    st.write("display matplotlib in stramlit: https://docs.streamlit.io/library/api-reference/charts/st.pyplot")
    st.write("streamlit stuff: https://canvas.eee.uci.edu/courses/39211/files/folder/Discussion%20Notebooks/Week%204?preview=16080226")
    st.write("kmeans： https://christopherdavisuci.github.io/UCI-Math-10/Week7/Week6-Friday.html")
    st.write("matplotlib bar graph: https://benalexkeen.com/bar-charts-in-matplotlib/")
    st.write("the project I referenced: https://towardsdatascience.com/lol-match-prediction-using-early-laning-phase-data-machine-learning-4c13c12852fa")
    st.write("pandas dataframe: https://stackoverflow.com/questions/10665889/how-to-take-column-slices-of-dataframe-in-pandas ")
    st.write("altair chart: https://altair-viz.github.io/gallery/scatter_tooltips.html")
    st.write("standardscalar: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html")
    st.write("training_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html")
