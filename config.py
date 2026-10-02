import os
from dotenv import load_dotenv


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


KAGGLE_DATASET_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
TARGET_FILE_NAME = "HI-Small_Trans.csv"
DATA_PATH = "HI-Small_Trans.csv"
SAMPLE_SIZE = 5000


CONTAMINATION_RATE = 0.01
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = 6
MAX_DAILY_ALERTS = 30
