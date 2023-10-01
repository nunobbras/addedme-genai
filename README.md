# DDChatbot

First create a `.env` file based on `.env.template`
with your own codes for openAI API.

Rebuild the docker image:

```
docker build -t addedmegenai:latest . 
```

Run the docker: 

```
docker run -p 5001:5000 addedmegenai:latest 
```

Access the endpoint with a GET here:

```
http://localhost:5001/
```

You should get `here API genAI!!`

To test the end point use the notebook `notebooks/usage_example_curl.ipynb`

Run it to see the post result - It should take about 50 seconds to get all needed data.

To see the class working use the notebook `notebooks/usage_example.ipynb`


Any question email me (nuno@daredata.engineering)