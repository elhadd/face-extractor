import time

# Simula un processo di elaborazione con output periodici di progresso
for i in range(1, 11):
    # Simula il tempo di elaborazione
    time.sleep(1)

    # Stampa il progresso
    print(f"Progresso: {i * 10}%")

# Simula il completamento del processo
print("Elaborazione completata.")
