# Roteiro sugerido (8–10 min)
1. Problema: mensagens fraudulentas e riscos de engenharia social (1 min).
2. Objetivo e limites: ferramenta educativa, não verificação definitiva (1 min).
3. Arquitetura: CSV → limpeza → tokenizer → Transformer local → embeddings → k-NN → decisão (2 min).
4. Justificativas: modelo multilíngue, normalização e k=3; sem fine-tuning porque configuração é permitida (1 min).
5. Colab: mostrar dataset separado, exemplos, matriz de confusão e métricas **reais** (2 min).
6. Deploy: digitar mensagem nova e mostrar sinais e exemplos semelhantes (2 min).
7. Limitações, ética, perguntas e conclusão (1 min).

Perguntas prováveis: Por que Hugging Face? O que são embeddings? Por que k=3? Como evitar vazamento de dados? A porcentagem é probabilidade? Por que não treinar do zero? O que acontece sem internet? O que torna a solução limitada?

Respostas centrais: embeddings representam significado em vetores; referências e teste são separados; porcentagem é índice de voto ponderado, não risco calibrado; modelo executa no runtime local após baixar pesos; conjunto sintético é apenas prova de conceito.
