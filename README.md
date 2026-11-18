---
title: CyberShield AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# CyberShield AI — Trabalho Final NLP (FECAF)

Aplicação didática que usa embeddings de um Transformer multilíngue do Hugging Face para comparar mensagens com exemplos rotulados e indicar possíveis sinais de golpes. **Não é um detector validado para uso real.**

## Arquivos
- `notebooks/CyberShield_AI_Colab.ipynb`: pipeline e avaliação no Colab
- `app.py`: interface Streamlit
- `model_utils.py`: pipeline compartilhado entre notebook e app
- `data/mensagens.csv`: 20 exemplos de referência e 8 de teste (sintéticos)
- `Dockerfile`: deploy Hugging Face Spaces Docker

## Executar localmente
Python 3.10 ou 3.11 recomendado.
```bash
pip install -r requirements.txt
streamlit run app.py
```
Primeira execução baixa pesos do Hugging Face; inferência roda no ambiente local. Requer internet para baixar pesos inicialmente.

## Google Colab
1. Envie a pasta inteira ao GitHub ou faça upload de `model_utils.py` e `data/mensagens.csv` no Colab mantendo a estrutura.
2. Abra `notebooks/CyberShield_AI_Colab.ipynb` no Colab.
3. Edite a variável `REPO_URL` na célula de configuração para a URL do seu repositório GitHub.
4. Execute todas as células e salve uma cópia com saídas no Drive.

## Deploy Hugging Face Spaces
1. Crie um Space novo com **Docker** (público).
2. Envie o conteúdo da raiz deste projeto ao Space (incluindo `Dockerfile`, `requirements.txt`, `app.py`, `model_utils.py`, `data/`).
3. Aguarde o status `Running`, abra o link e teste mensagens diferentes.
4. Se houver erro, consulte `Logs` no Space e corrija antes da apresentação.

## Decisões técnicas
- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, suporta português e embeddings semânticos.
- `k=3`: voto de três vizinhos para reduzir dependência de um exemplo isolado; compare com k=1,5,7 na avaliação.
- Normalização L2: produto escalar equivale à similaridade de cosseno.
- Limpeza mínima: preserva sinais textuais; tokenizer é o do modelo.
- Não há fine-tuning: a exigência permite **configurar** o modelo. Dados são protótipos sintéticos, não coleta real.
- Índice calculado não é probabilidade calibrada nem prova de fraude.

## Limitações
Amostra pequena, artificial e possivelmente enviesada; resultados não generalizam. Ataques novos, mensagens ambíguas e linguagem informal podem falhar. Não inclua dados pessoais reais nas demonstrações.
