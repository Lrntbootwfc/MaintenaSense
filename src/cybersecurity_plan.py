import hashlib

def hash_file(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"SHA-256 hash of {file_path}: {file_hash}")

if __name__ == "__main__":
    hash_file("data/processed_data.csv")
