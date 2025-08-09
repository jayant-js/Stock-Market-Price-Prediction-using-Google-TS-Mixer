from fastapi import FastAPI, Path, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import sys
from threading import Thread
from pipeline.pipeline_module import create_prediction_pipeline, gather_ticker_data, encode_year

def load_pipeline():
    app.state.pipe = create_prediction_pipeline()

@asynccontextmanager
async def lifespan(app:FastAPI):
    sys.modules["__main__"].encode_year = encode_year # type: ignore
    Thread(target=load_pipeline).start()
    yield

app = FastAPI(lifespan=lifespan)

@app.get('/')
def intro():
    return {'message':'This is Stock Price Prediction API'}

@app.get('/predict/{ticker}')
def get_predictions(ticker = Path(..., description='Give the Ticker Symbol here')):
    try:
        new_data = gather_ticker_data(ticker_symbol=ticker)
        pipeline = app.state.pipe
        forecast = pipeline.predict(new_data, forecast_horizon = 7)
        response = {
            'ticker':ticker,
            'forecast':forecast.to_dict('records'),
            'historical':new_data.tail(30).reset_index().to_dict('records')
        }
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 