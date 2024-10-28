from featureform import Client

# Konfiguracja klienta Featureform
client = Client(insecure=True)

# Nazwy zestawu treningowego i wariantu
ts_name = "llm_training"
ts_variant = "bsa"

# Pobranie zestawu treningowego
dataset = client.training_set(ts_name, ts_variant)

# Iteracja przez zestaw treningowy i wyświetlenie danych
for i, batch in enumerate(dataset):
    print(f"Batch {i}: {batch}")
