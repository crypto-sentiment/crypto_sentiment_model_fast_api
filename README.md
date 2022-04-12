# Cryptonews sentiment classification model

This is a simple cryptonews sentiment classification Fast API.

<img src='static/img/fast_api_screenshot.png' width=600>

## Setting up the environment

To reproduce model training and running the API, perform the following steps:

 - install [Poetry](https://python-poetry.org/docs/) with Python >= 3.8
 - run `poetry install` to set up a virtual environment for the project and install all dependencies
 - (opt.) `poetry run pytest`

## Data

The training set is comprised of ~4500 news titles manually classiified into positive, neutral or negative. The data is not shared but you can refer to a toy dataset with 50 records: `data/toy_train.csv`.

```
title,sentiment
Bitcoin Market Has Run Out of Juice: Cryptocurrency Analyst,Negative
Bitcoin Core 0.14.0 Speeds Up Blockchain Syncing By 48%,Positive
Thinking of Travelling With Bitcoin? With These Websites You Can,Positive
```

## Model

The model is a simple Tf-Idf & logreg Sklearn [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). To reproduce model training, perform the following steps:

1. Check project configurations defined in `config.yaml`, e.g. model parameters and path to the model, etc.
2. Run `poetry run python -m cryptonews_sentiment_base_model.train` (might take a while for the first time, even with a small dataset)

## API

Here we are using Fast API, [first steps](https://fastapi.tiangolo.com/tutorial/first-steps/).

To run the API locally: ```poetry run uvicorn cryptonews_sentiment_base_model.api:app --reload```

Then you can talk to the `http://127.0.0.1:8000/classify` endpoint sending text data as a dictionary with text field as a key and arbitrary text as a value, e.g.

```
{
   "title": "BTC drops by 10% this Friday"
}
```

Example of a Curl request:

```
curl --location --request POST 'http://127.0.0.1:8000/classify' \
--header 'Content-Type: application/json' \
--data-raw '{ "title": "BTC drops by 10% this Friday" }'
```


### API docs

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Build docs

`poetry run sphinx-build -a -b html docs/ docs/_build`
