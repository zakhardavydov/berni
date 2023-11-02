# BERNI

## Get started

### Using ``poetry`` 

Install `poetry`:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Run a shell in the project's environment:
```bash
poetry shell
```

Run `main.py` to test the setup:
```bash
python play.py
```

### MLFlow config (Set up local server)
``Optional if using non-local server``

Start MLFlow
```bash
docker-compose -f docker-compose.mlflow.yaml up
```

### Configure .env file
When first time cloning the repository, create a ``.env`` file under ``Group_Research_Project_G10``, add belowing content to the file. Then,
``replace`` all ``$...$`` part with your MLflow server setting
```
ML_FLOW_URL=$YOUR ML_FLOW SERVER URL$
AWS_ACCESS_KEY_ID=$YOUR AWS_ACCESS_KEY_ID$
AWS_SECRET_ACCESS_KEY=$YOUR AWS_SECRET_ACCESS_KEY$
MLFLOW_S3_ENDPOINT_URL=$YOUR S3_ENDPOINT_URL$
CURRENT_USER_DIR=$YOUR USER DIRECTORY
```
