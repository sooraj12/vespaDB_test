
<!-- Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

# Vespa sample applications

A simple Vespa application which can be deployed on one node,

Follow [Vespa getting started](https://cloud.vespa.ai/en/getting-started) to deploy this.

This repo is based on (https://blog.vespa.ai/turbocharge-rag-with-langchain-and-vespa-streaming-mode/)

```
python create.py # create vespa application package and deploy to docker container
python loader.py # load embeddings from pdf

python langchain_retriever.py # langchain BaseRetriever with vespa
```
