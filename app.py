import streamlit as st
import requests
import json


def main():
    st.title("API Frontend - POST-GET Debugger")
    url_API = st.text_input("inserisci url dell'api",
                            "http://localhost:8001/predict")
    rd = st.number_input("Inserisci rd", 0, 1000000, 73721)
    admin = st.number_input("Inserisci admin", 0, 1000000, 121344)
    market = st.number_input("Inserisci market", 0, 1000000, 211025)

    ############## GET REQUEST #################
    if st.button("Predict with GET"):
        url = url_API
        url2 = f"?rd={rd}&admin={admin}&market={market}"
        link = url+url2
        st.write('"{}"'.format(link))
        response = requests.get(link)
        result = response.json()
        st.success(f"The result is: {result['prediction']}")

    ############## POST REQUEST #################
    if st.button("Predict with POST"):
        url = url_API
        response = requests.post(url,
                                 headers={"Content-Type": "application/json"},
                                 data=json.dumps({
                                     "rd": rd,
                                     "admin": admin,
                                     "market": market,
                                 })
                                 )
        result = response.json()
        st.success(f"The result is: {result['prediction']}")


if __name__ == '__main__':
    main()
