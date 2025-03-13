# Object-Detection-with-YOLO

API em python utilizando FastAPI
Processamento de Imagens com YOLO

"""
## Instruções para configurar o ambiente e executar a aplicação

### 1. Criar e ativar o ambiente virtual
```bash
python -m venv venv  # Criar ambiente virtual
source venv/bin/activate  # Ativar no Linux/macOS
venv\Scripts\activate  # Ativar no Windows
```

### 2. Instalar dependências
é sempre bom veriricar
```bash
python.exe -m pip install --upgrade pip
```

Back-end
```bash
pip install torch fastapi uvicorn opencv-python pillow numpy ultralytics
```

### 3. Entrar na pasta da frontend
```bash
cd frontend
```


### 5. Entrar na pasta da API
```bash
cd backend
pip install "fastapi[standard]"
```

### 6. Executar a API
```bash
venv\Scripts\activate
fastapi dev backend.py
```

A API estará disponível em `http://127.0.0.1:8000/docs` para testes via Swagger UI.
"""
