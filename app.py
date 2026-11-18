import streamlit as st
from model_utils import load_data, load_model, build_index, predict, evaluate

st.set_page_config(page_title="CyberShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberShield AI")
st.caption("Análise educativa de mensagens por NLP local — não substitui verificação de segurança")
@st.cache_resource(show_spinner="Carregando modelo Hugging Face local...")
def resources():
    df = load_data()
    model = load_model()
    ref, embeddings = build_index(model, df)
    return df, model, ref, embeddings
try:
    df, model, ref, embeddings = resources()
except Exception as exc:
    st.error(f"Não foi possível carregar o modelo: {exc}")
    st.stop()
with st.sidebar:
    st.header("Sobre")
    st.write("Modelo: paraphrase-multilingual-MiniLM-L12-v2")
    st.write("Técnica: embeddings + k vizinhos por similaridade de cosseno")
    k = st.slider("Vizinhos (k)", 1, 7, 3, 2)
    st.warning("O índice não é probabilidade de fraude; não clique em links suspeitos.")
tab1, tab2, tab3 = st.tabs(["🔎 Analisar", "📊 Avaliação", "📘 Como funciona"])
with tab1:
    sample = st.selectbox("Exemplos", ["Digite sua mensagem", "URGENTE! Sua conta será bloqueada. Envie seu código SMS agora.", "A reunião do grupo será amanhã na biblioteca."])
    value = "" if sample == "Digite sua mensagem" else sample
    message = st.text_area("Cole a mensagem", value=value, height=145)
    if st.button("Analisar mensagem", type="primary"):
        if not message.strip():
            st.warning("Digite uma mensagem para analisar.")
        else:
            result = predict(message, model, ref, embeddings, k)
            st.subheader("Possivelmente suspeita ⚠️" if result["rotulo"] == "suspeita" else "Semelhança com mensagens legítimas ℹ️")
            st.metric("Índice de similaridade com a classe suspeita", f'{result["indice"]:.1%}')
            st.write("**Sinais textuais:**", ", ".join(result["sinais"]) if result["sinais"] else "Nenhum sinal da lista encontrado")
            st.write("**Exemplos semelhantes usados na decisão:**")
            for ex in result["exemplos"]:
                st.write(f'• [{ex["rotulo"]}] {ex["texto"]} — similaridade: {ex["similaridade"]:.3f}')
            st.info("Uma mensagem sem sinais óbvios também pode ser golpe. Confirme sempre pelo canal oficial.")
with tab2:
    metrics = evaluate(model, ref, embeddings, df, k)
    st.metric("Acurácia no pequeno conjunto de teste", f'{metrics["accuracy"]:.1%}')
    st.write("**Matriz de confusão** (linhas: real; colunas: previsto)")
    st.dataframe(metrics["confusion"], use_container_width=True)
    st.dataframe(metrics["test"].assign(previsao=metrics["preds"]), use_container_width=True)
    st.caption("Amostra didática pequena e sintética: resultados não representam desempenho no mundo real.")
with tab3:
    st.markdown("""1. Carrega dados rotulados de referência e teste.
2. Limpa espaços e tokeniza com o tokenizer do Transformer.
3. Gera embeddings no próprio runtime, sem APIs externas de IA.
4. Compara a mensagem aos exemplos por similaridade de cosseno.
5. Vota entre os k exemplos mais próximos e apresenta evidências.
6. Avalia em exemplos separados da referência.""")
