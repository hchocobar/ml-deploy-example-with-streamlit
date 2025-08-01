from pickle import load
import streamlit as st


model = load(open('iris-decision-tree-classifier-default-18.pkl', 'rb'))
class_dict = {"0": "Iris setosa",
              "1": "Iris versicolor",
              "2": "Iris virginica"}

st.title("Iris - Model prediction")

val1 = st.slider("Petal width", min_value = 0.1, max_value = 2.5, step = 0.1)
val2 = st.slider("Petal length", min_value = 1.0, max_value = 6.9, step = 0.1)
val3 = st.slider("Sepal width", min_value = 2.0, max_value = 4.4, step = 0.1)
val4 = st.slider("Sepal length", min_value = 4.3, max_value = 7.9, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)

st.subheader('Power by:')
st.markdown("""[Héctor Chocobar Torrejón](https://chocobar.net)""")
