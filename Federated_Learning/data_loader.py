import os, requests, zipfile, io

def download(root: str = "data") -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "Dataset/tiny_shakespeare.txt")
    if not os.path.exists(path):
        print("Downloading tiny‑Shakespeare …")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        txt = requests.get(url, timeout=30).text
        with open(path, "w", encoding="utf8") as f:
            f.write(txt)
    
    print("Dataset loaded successfully.")
    return path