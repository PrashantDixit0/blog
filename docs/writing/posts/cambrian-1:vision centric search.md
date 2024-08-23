---
draft: False
date: 2024-08-13
slug: cambrian-1-vision-centric-search
comments: True
categories:
  - Multimodal LLM
authors:
  - Prashant
---

# Cambrian-1: Vision-Centric Search

Cambrian-1 is a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. While stronger language models can boost multimodal capabilities, the design choices for vision components are often insufficiently explored and disconnected from visual representation learning research.

![alt text](img/cambrian/cambrian-result-1.png)

Cambrian-1 is built on five key pillars, each providing important insights into the design of multimodal LLMs (MLLMs):

1. ***Visual Representations***: They explore various vision encoders and their combinations.
2. ***Connector Design***: They design a new dynamic, spatially-aware connector that integrates visual features from several models with LLMs while reducing the number of tokens.
3. ***Instruction Tuning Data***: They curate high-quality visual instruction-tuning data from public sources, emphasizing distribution balancing.
4. ***Instruction Tuning Recipes***: They discuss strategies and best practices for instruction tuning.
5. ***Benchmarking***: They examine existing MLLM benchmarks and introduce a new vision-centric benchmark called "CV-Bench".

We'll learn how Cambrian-1 works with an example of Vision-Centric Exploration on images found through vector search. This will involve two steps.
1. Performing vector search to get related images
2. Use obtained images for vision-centric exploration

<!-- more -->

![alt text](./img/cambrian/cambrian-result-1.png)


## Implementation
This blog contains code snippets, for the whole code with their description use checkout Kaggle Notebook and run it with your own prompt

!!! Note "Kaggle Notebook"
    Kaggle notebook demonstrating this tutorial

    https://www.kaggle.com/code/prasantdixit/cambrian-1-vision-centric-exploration-of-images/ 

In this example, We will be working with the Flickr-8k dataset. It is a multi-modal dataset comprising images and their corresponding captions.

We'll index these images based on their captions using the all-mpnet-base-v2 model from the sentence-transformer. The LanceDB Embedding API will retrieve embeddings from the sentence transformer models. 

```python
embedding_model = (
    get_registry()
    .get("sentence-transformers")
    .create(name="all-mpnet-base-v2", device="cuda:0")
)
```

Now we are ready to integrate this embedding function into the schema of the table to simplify the process of data ingestion and querying.

```python
pa_schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 768)),
    pa.field("image_id", pa.string()),
    pa.field("image", pa.binary()),
    pa.field("captions",pa.string()),
])

class Schema(LanceModel):
    vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()
    image_id: str
    image: bytes
    captions: str = embedding_model.SourceField()
```

## Ingestion Pipeline
Storing vector embeddings efficiently is crucial for leveraging their full potential in various applications. To manage, query, and retrieve embeddings effectively, especially with large-scale and multi-modal data, we need a robust solution.
LanceDB allows you to store, manage, and query embeddings as well as raw data files.

```python
db = lancedb.connect("dataset")
tbl = db.create_table("table", schema=Schema, mode="overwrite")
tbl.add(process_dataset(df))
```

The above snippet populates a table that you can use to query using captions.

## Vector Search
With LanceDB, performing vector search is pretty straightforward. LanceDB Embedding API implicitly converts the query into embedding and performs vector search to give desired results. Let's take a look at the following example:

```python
query = "cat sitting on couch"

hit_lists = tbl.search(query) \
    .metric("cosine") \
    .limit(2) \
    .to_list()

for hit in hit_lists:
    show_image(hit["image"])
```

This code sample returns the top two similar images, you can increase the number by changing `limit` parameter. 

The results would look like this

![alt text](./img/cambrian/cambrian-result-2.png)

![alt text](./img/cambrian/cambrian-result-3.png)

Now we have obtained images from the vector search, let's explore these images with a vision-centric multi-modal LLM(MLLM) Cambrian-1.

## Setup vision-centric Exploration
We'll use Cambrian-1, a family of multimodal LLMs (MLLMs) designed with a vision-centric approach. For setting up Cambrian-1, we'll use their official GitHub repo https://github.com/cambrian-mllm/cambrian. 


Clone the repo and install requirements 
```python
# clone 
!git clone https://github.com/cambrian-mllm/cambrian
%cd cambrian

# install gpu related requirements
!pip install ".[gpu]" -q
```

!!! Note "Kaggle Notebook"
    To go through this notebook, run each cell one by one.

    https://www.kaggle.com/code/prasantdixit/cambrian-1-vision-centric-exploration-of-images/ 

```python
prompt = "How many cats are in image and Why?"
images_path = []
for hit in hit_lists:
    image_path = f"/kaggle/working/Images/{hit['image_id']}"
    images_path.append(image_path)
    
infer_cambrian(images_path, prompt)

import subprocess
command = "python3 inference.py"
output = subprocess.check_output(command, shell=True, text=True)
```

This code snippet will run `inference.py` and return the query results corresponding to each input image.

The result will look like this

![alt text](./img/cambrian/cambrian-result-4.png)

Our Prompt for exploring images was `How many cats are in image and Why?` , and Cambrian-1 has clearly given the right and clear output stating the correct count and external scenario.

## Conclusion
In this tutorial, We've explored the field of vision-centric exploration same as GPT4-o where you give an image as input and ask questions related to it. We have utilized Cambrian-1 as a multi-modal LLM for vision-centric exploration combined with vector search using text embeddings.

We've seen how LanceDB, with its powerful and elegant design in Embedding API, simplifies the process of storing, managing, and querying these embeddings, making sophisticated tasks like semantic search both accessible and efficient.