import json
import os

from superlinked import framework as sl


class Product(sl.Schema):
    id: sl.IdField
    description: sl.String
    rating: sl.Integer


product = Product()

description_space = sl.TextSimilaritySpace(
    text=product.description, model="Alibaba-NLP/gte-large-en-v1.5"
)
rating_space = sl.NumberSpace(
    number=product.rating, min_value=1, max_value=5, mode=sl.Mode.MAXIMUM
)
index = sl.Index([description_space, rating_space], fields=[product.rating])


# Define your query and parameters to set them directly at query-time
# or let an LLM fill them in for you using the `natural_language_query` param.
# Don't forget to set your OpenAI API key to unlock this feature.
query = (
    sl.Query(
        index,
        weights={
            description_space: sl.Param("description_weight"),
            rating_space: sl.Param("rating_weight"),
        },
    )
    .find(product)
    .similar(
        description_space,
        sl.Param(
            "description_query",
            description="The text in the user's query that refers to product descriptions.",
        ),
    )
    .limit(sl.Param("limit"))
    .with_natural_query(
        sl.Param("natural_language_query"),
        sl.OpenAIClientConfig(api_key=os.environ["OPEN_AI_API_KEY"], model="gpt-4o")
    )
)

# Run the app in-memory (server & Apache Spark executors available too!).
source = sl.InMemorySource(product)
executor = sl.InMemoryExecutor(sources=[source], indices=[index])
app = executor.run()


# Ingest data into the system - index updates and other processing happens automatically.
source.put([
    {
        "id": 1,
        "description": "Budget toothbrush in black color. Just what you need.",
        "rating": 1,
    },
    {
        "id": 2,
        "description": "High-end toothbrush created with no compromises.",
        "rating": 5,
    },
    {
        "id": 3,
        "description": "A toothbrush created for the smart 21st century man.",
        "rating": 3,
    },
])

result = app.query(query, natural_query="best toothbrushes", limit=1)

# Examine the extracted parameters from your query
print(json.dumps(result.knn_params, indent=2))

# The result is the 5-star rated product.
result.to_pandas()