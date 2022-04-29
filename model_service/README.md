# Model service

## Structure
```
/app/
/conf/
/trained_models/
    ├── logit_tfidf_btc_sentiment.pkl
    └── bert.ckpt
```
checkpoints can be downloaded here: https://drive.google.com/drive/folders/1zqfxJBc3F2Wx2lkbE755XZE_wSFR9069?usp=sharing

## Run
 Tested with docker v20.10.14 and docker-compose v2.2.3.

 Download and place checkpoints in trained_models folder.
 Run:

```
docker compose up --build
```
Check http://localhost:8001/docs
