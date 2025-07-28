from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi import Query, Path
import pandas as pd
import joblib
from pipeline.pipeline_module import create_pipeline, DartsModelWrapper

app = FastAPI()

@app.get('/intro')
def intro():
    return {"Message":"Hey this is just a introduction page for stock market price prediction"}

@app.post('/get-ticker/{ticker}')
def get_ticker(ticker: str = Path(..., description="This is just for an example")):
    try:
        pipeline = create_pipeline(ticker=ticker, forecast_horizon=7)
        pipeline.fit(X = pd.DataFrame())
        model = pipeline.named_steps['model_training_and_prediction']
        model.save_model('stock_prediction_model.pt')
        return {'message':f'Pipeline trained and saved for ticker {ticker}'}
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))

@app.get('/get-ticker/prediction')
def ticker_prediction():
    try:
        model_params = {
            'input_chunk_length': 14,
            'ff_size': 9,
            'num_blocks': 8,
            'hidden_size': 5,
            'n_epochs': 7,
            'use_reversible_instance_norm': True
        }
        model_wrapper = DartsModelWrapper(model_params=model_params, output_chunk_length=7)
        model = model_wrapper.load_model(path = 'stock_prediction_model.pt')
        forecast_df = model.predict(n = 7)
        forecast_df = forecast_df.rename_axis("Date").reset_index()
        forecast_df["Date"] = forecast_df['Date'].astype(str)
        forecast_records = forecast_df.to_dict(orient='records')
        return JSONResponse(content={'forecast':forecast_records})  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))