# DDChatbot

## Setup

Before using this repo install all the requirements

```shell
pip install -r requirements.txt
```


Then install the package

```shell
pip install .
```

For **development** install in editable mode


```shell
pip install -e .
```

## Starting flask server

To use the created flask api first create a `.env` file with the following variables

```bash
AZURE_STORAGE_ACCOUNT=
AZURE_STORAGE_CONTAINER= 
AZURE_SEARCH_SERVICE= 
AZURE_SEARCH_INDEX= 
AZURE_OPENAI_SERVICE= 
AZURE_OPENAI_GPT_DEPLOYMENT=text-davinci-003
AZURE_OPENAI_CHATGPT_DEPLOYMENT=gpt-35-turbo
OPENAI_API_KEY= 
DB_CONNECTION_STRING=mongodb://localhost:27017/<db-name>
DB_NAME=
```

then run 

```` bash
python app.py
```

