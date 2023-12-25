#importing the required libraries
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

#importing the ticker CSV file
dataframe=pd.read_csv("ticker and comapny list.csv")
ticker=dataframe.drop('Company Name',axis=1)
companylist=dataframe.drop('Symbol',axis=1)
ticker=ticker.values.tolist()
updated_ticker=[]
for sample in ticker:
  sample1=' '.join(str(e)for e in sample)
  updated_ticker.append(sample1)
companylist=companylist.values.tolist()
updated_company=[]
for sample in companylist:
  sample1=' '.join(str(e)for e in sample)
  updated_company.append(sample1)
#creating the start and end date
start = "2015-01-01"
today=date.today()

#creating the title and the sidebarime

st.title("Find Stock Company here")
company=st.selectbox("Select Company Name",updated_company)
st.title("Stock Trend Prediction")



selected_stock = st.selectbox("Select dataset for prediction",updated_ticker)
n_years = st.slider("Years of prediction:",0,4)
n_week =st.slider("weeks of prediction:",0,52)
n_day=st.slider("Days of prediction:",0,30)
period=(n_years*365)+(n_week*7)+(n_day)
@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data
data_load_state = st.text("Load data...")
data= load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

#ploting the raw data   
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#forecasting the data
df_train=data[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('Forecast data')
st.write(forecast.tail(100))

#ploting the forecast data
st.write("Forecast data")
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)