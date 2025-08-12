# Currency_Exchange_Rate_Predict
Objective: This project is built to predict currency exchange rate together with LLM models to explain the results. 

1. Exchange_Rate_CAD_CNY_Forecast.ipynb - a simple demo that is using Prophet to predict 14 days of currency exchange rate, and meta-llama/Llama-2-7b-chat-hf model (from Hugging Face) to explain insights of the predicted results.

*Next Step*:
- For forecasting: 
  - Use ARIMA to replace prophet and compare the results
  - Use LSTM to replace prophet and compare the results
 
- For LLM:
  - Apply RAG + other prompts techniques
  - Try Google Gemini model
  - Add evaluation codes
 
- For deployment:
  - Migrate from jupyter/colab notebook to app ready codes
  - Deploy on cloud servers
  - Introduce automation of running modeling
  - Introduce auto-evaluation 
