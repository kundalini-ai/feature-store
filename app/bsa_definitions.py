import featureform as ff

# Rejestracja PostgreSQL
postgres = ff.register_postgres(
    name="postgres-bsa",
    host="host.docker.internal",  # Nazwa DNS dockera dla PostgreSQL
    port="5432",
    user="postgres",
    password="password",
    database="postgres",
)

# Rejestracja Redis
redis = ff.register_redis(
    name="redis-bsa",
    host="host.docker.internal",  # Nazwa DNS dockera dla Redis
    port=6379,
)

# Rejestracja tabeli w PostgreSQL
llm_data = postgres.register_table(
    name="llm_data",
    table="llm_data",  # Nazwa tabeli w PostgreSQL
)

# Transformacje SQL dla kaÅ¼dej cechy
@postgres.sql_transformation(inputs=[llm_data])
def extract_question(ld):
    return "SELECT id as data_id, question from {{ld}}"

@postgres.sql_transformation(inputs=[llm_data])
def extract_context(ld):
    return "SELECT id as data_id, context from {{ld}}"

@postgres.sql_transformation(inputs=[llm_data])
def extract_response(ld):
    return "SELECT id as data_id, response from {{ld}}"

# Definicja encji LLMData
@ff.entity
class LLMData:
    question = ff.Feature(
        extract_question[["data_id", "question"]],
        variant="bsa",
        type=ff.String,
        inference_store=redis,
    )

    context = ff.Feature(
        extract_context[["data_id", "context"]],
        variant="bsa",
        type=ff.String,
        inference_store=redis,
    )

    response = ff.Feature(
        extract_response[["data_id", "response"]],
        variant="bsa",
        type=ff.String,
        inference_store=redis,
    )

# Rejestracja zestawu treningowego
ff.register_training_set(
    name="llm_training",
    label=LLMData.category,
    features=[LLMData.question, LLMData.context, LLMData.response],
    variant="bsa",
)