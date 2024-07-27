from vespa.package import (
    Schema,
    Document,
    Field,
    FieldSet,
    HNSW,
    ApplicationPackage,
    Component,
    Parameter,
    RankProfile,
    Function,
    FirstPhaseRanking,
)
from vespa.deployment import VespaDocker

app_name = "pdfragagent"
schema_name = "pdf"

# define schema
pdf_schema = Schema(
    name=schema_name,
    mode="streaming",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "index"]),
            Field(name="title", type="string", indexing=["summary", "index"]),
            Field(name="url", type="string", indexing=["summary", "index"]),
            Field(name="authors", type="array<string>", indexing=["summary", "index"]),
            Field(name="page", type="int", indexing=["summary", "index"]),
            Field(
                name="metadata",
                type="map<string,string>",
                indexing=["summary", "index"],
            ),
            Field(name="chunks", type="array<string>", indexing=["summary", "index"]),
            Field(
                name="embedding",
                type="tensor<bfloat16>(chunk{}, x[384])",
                indexing=["input chunks", "embed e5", "attribute", "index"],
                ann=HNSW(distance_metric="angular"),
                is_document_field=False,
            ),
        ]
    ),
    fieldsets=[FieldSet(name="default", fields=["chunks", "title"])],
)

# define application
application_package = ApplicationPackage(
    name=app_name,
    schema=[pdf_schema],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"
                    },
                ),
            ],
        )
    ],
)

# define ranking
semantic = RankProfile(
    name="hybrid",
    inputs=[("query(q)", "tensor<float>(x[384])")],
    functions=[
        Function(
            name="similarities",
            expression="cosine_similarity(query(q), attribute(embedding),x)",
        )
    ],
    first_phase=FirstPhaseRanking(
        expression="nativeRank(title) + nativeRank(chunks) + reduce(similarities, max, chunk)",
        rank_score_drop_limit=0.0,
    ),
    match_features=[
        "closest(embedding)",
        "similarities",
        "nativeRank(chunks)",
        "nativeRank(title)",
        "elementSimilarity(chunks)",
    ],
)

pdf_schema.add_rank_profile(semantic)

# deploy application to vespa container
# container_name = "vespaDB"
# vespa_docker = VespaDocker.from_container_name_or_id(container_name)
vespa_docker = VespaDocker()
vespa_docker.deploy(application_package=application_package)
