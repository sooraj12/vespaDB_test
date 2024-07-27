import json
from vespa.io import VespaQueryResponse
from app import app

# file to test queries
response: VespaQueryResponse = app.query(
    yql="select id,title,page,chunks from pdf where userQuery() or ({targetHits:10}nearestNeighbor(embedding,q))",
    groupname="test_user",
    ranking="hybrid",
    query="why is colbert effective?",
    body={
        "presentation.format.tensors": "short-value",
        "input.query(q)": 'embed(e5, "why is colbert effective?")',
    },
)

assert response.is_successful()
print(json.dumps(response.hits[0], indent=2))
