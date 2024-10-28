from featureform import Client

# Konfiguracja klienta Featureform
serving = Client(insecure=True)

# Nazwy transformacji i wariantów
question_transformation = "question"
context_transformation = "context"
response_transformation = "response"
category_transformation = "category"
transformation_variant = "bsa"

# Przykładowy identyfikator danych
data_id = "1"

# Debugowanie: Wyświetlenie informacji o transformacjach
print(f"Fetching features for data_id: {data_id}")
print(f"Transformations: {question_transformation}, {context_transformation}, {response_transformation}, {category_transformation}")
print(f"Variant: {transformation_variant}")

# Pobranie cech dla danego identyfikatora
try:
    features = serving.features(
        [
            (question_transformation, transformation_variant),
            (context_transformation, transformation_variant),
            (response_transformation, transformation_variant),
            (category_transformation, transformation_variant),
        ],
        {"id": data_id}
    )
    # Wyświetlenie wyników
    print(f"Features for data_id {data_id}: {features}")
except Exception as e:
    print(f"An error occurred: {e}")
