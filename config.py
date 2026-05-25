import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
MODELS_COMPARISON_DIR = os.path.join(PROJECT_DIR, 'models_comparison')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_COMPARISON_DIR, exist_ok=True)

# Pliki z danymi
KPUU_FILE = os.path.join(DATA_DIR, 'train_kpuu.csv')
KP_FILE = os.path.join(DATA_DIR, 'train_kp.csv')
KPUU_LOG_FILE = os.path.join(DATA_DIR, 'train_kpuu_log.csv')
KP_LOG_FILE = os.path.join(DATA_DIR, 'train_kp_log.csv')

# Parametry modelu
NODE_FEATURES = 74   # standardowa liczba cech atomów
EDGE_FEATURES = 12   # standardowa liczba cech wiązań
HIDDEN_DIM = 128
GROWTH_RATE = 32

# Parametry treningu
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.0002
EARLY_STOP = 200
