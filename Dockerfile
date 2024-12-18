FROM deepset/haystack:base-main

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

RUN pip install jsonschema python-dotenv fastapi uvicorn
RUN pip install git+https://github.com/deepset-ai/haystack-experimental@main

COPY . /app

EXPOSE 1416

CMD uvicorn utils.fast_api:app --host 0.0.0.0 --port 1416