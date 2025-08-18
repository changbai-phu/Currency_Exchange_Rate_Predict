# Currency Exchange Rate Predict
Objective: This project is built to predict currency exchange rate together with LLM models to explain the results. 

1. Exchange_Rate_CAD_CNY_Forecast.ipynb - a simple demo that is using Prophet to predict 14 days of currency exchange rate, and meta-llama/Llama-2-7b-chat-hf model (from Hugging Face) to explain insights of the predicted results.
2. Different_Forecast.ipynb - Use ARIMA and LSTM to predict 14 days of exchange rate, compare predicted data generated from Prophet, ARIMA and LSTM, compare and evaluate predicted data with actual 14 exchange rates. As a result, Prophet performs the best among the 3 models based on MAE and RMSE. 

*Next Step*:
- For forecasting: 
  - Use ARIMA to replace prophet and compare the results (done)
  - Use LSTM to replace prophet and compare the results (done)
 
- For LLM:
  - Apply RAG + other prompts techniques
  - Try Google Gemini model
  - Add evaluation codes
 
- For deployment:
  - Migrate from jupyter/colab notebook to app ready codes
  - Deploy on cloud servers
  - Introduce automation of running modeling
  - Introduce auto-evaluation 

### Lessons Learnt
Prophet
- additive time series model
- Good for seasonality 
- Robust to missing data
- Not good with sudden market shocks, less flexible for nonlinear data
- Use case: daily/weekly/monthly trends, web traffic, sale forecasting

LSTM
- Deep learning/RNN
- Captures long-term dependencies and nonlinear relationship
- Good for volatile and sequential data
- Requires lot of data and computation, sensitive to hyper-parameters
- Use case: complex time series, stock prices 

ARIMA
- autoRegressive Integrated Moving Average
- good for stationary non-strong seasonal patterns data, short-term forecasting
- need to find the best ARIMA order (p, d, q)
  - Using PACF to find p, ADF to choose d, ACF to find q

### Challenges
- By using the best ARIMA order calculated, the forecast becomes flat for the last several days.
  - have to try different way of finding order, like BIC instead of AIC used originally
  - try cross validation as well
  - but overall performance still does not perform really well.
