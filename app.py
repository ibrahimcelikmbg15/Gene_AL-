import os
import glob
import numpy as np
import faiss
import streamlit as st
from google import genai

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Gene-A(L)İ | Genom Okulu Asistanı",
    page_icon="🧬",
    layout="wide"
)

# --- SİBERPUNK TEMA (CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #39ff14 !important;
        font-family: 'Courier New', monospace;
    }
    .stChatMessage {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
    }
    .stSidebar {
        background-color: #010409;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown("### 🧬 Gene-A(L)İ v1.0")
    st.caption("Genom Okulu: 10. Kromozom Yapay Zekâ Asistanı")
    st.divider()

    # Streamlit Cloud'da secrets kullanır, yerelde kutucuktan girilir
    api_key = st.secrets.get("GEMINI_API_KEY", None) if "GEMINI_API_KEY" in st.secrets else None
    if not api_key:
        api_key = st.text_input("Gemini API Anahtarı", type="password", placeholder="AIzaSy...")

    st.divider()
    st.markdown("**📚 Yüklü Ders Transkriptleri:**")
    txt_files = sorted(glob.glob("transkriptler/*.txt"))
    if txt_files:
        for f in txt_files:
            st.markdown(f"- `{os.path.basename(f)}`")
    else:
        st.warning("`transkriptler/` klasöründe henüz .txt dosyası yok.")

# --- METİN PARÇALAMA VE FAISS İNDEKSLEME ---
def chunk_text(text, source_name, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source_name})
        start += chunk_size - overlap
    return chunks

@st.cache_resource(show_spinner="Gene-A(L)İ ders transkriptlerini hafızasına alıyor...")
def build_vector_store(_api_key, files):
    client = genai.Client(api_key=_api_key)
    all_chunks = []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        source = os.path.basename(filepath).replace(".txt", "")
        all_chunks.extend(chunk_text(content, source))

    if not all_chunks:
        return None, []

    texts = [c["text"] for c in all_chunks]
    embeddings = []

    # 50'şerli paketler halinde embedding al
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=batch
        )
        for emb in res.embeddings:
            embeddings.append(emb.values)

    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index, all_chunks

# --- ANA EKRAN ---
st.title(">_ Gene-A(L)İ | Genom Okulu")
st.caption("Biyoinformatik, SSR Markör Tasarımı ve Moleküler Islah sorularını sorabilirsin.")

if not api_key:
    st.info("Başlamak için sol menüye ücretsiz Google Gemini API anahtarını gir (aistudio.google.com).")
    st.stop()

if not txt_files:
    st.warning("`transkriptler` klasörüne en az bir `.txt` ders metni ekleyip sayfayı yenile.")
    st.stop()

index, chunks = build_vector_store(api_key, txt_files)
client = genai.Client(api_key=api_key)

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben Gene-A(L)İ. 🧬 Genom Okulu derslerinden aklına takılanları sor, siyah ekrandan korkma!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı sorusu
if prompt := st.chat_input("Örn: GMATA'da mononükleotitleri neden eliyoruz?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Genom taranıyor..."):
            # Soruyu vektöre çevir ve en alakalı 4 parçayı bul
            q_emb = client.models.embed_content(
                model="text-embedding-004",
                contents=prompt
            ).embeddings[0].values
            q_vec = np.array([q_emb], dtype="float32")
            faiss.normalize_L2(q_vec)

            _, indices = index.search(q_vec, k=min(4, len(chunks)))
            retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]

            context = "\n\n".join(
                [f"[Kaynak Ders: {c['source']}]\n{c['text']}" for c in retrieved]
            )

            system_prompt = f"""
Sen 'Gene-A(L)İ'sin; Doç. Dr. İbrahim Çelik'in kurduğu 'Genom Okulu: 10. Kromozom' platformunun yapay zekâ asistanısın.
Görevin: Öğrencilerin sorularını "Halk Ekmek" sadeliğinde ama akademik derinlikte, samimi bir dille yanıtlamak.

Kurallar:
1. Öncelikle aşağıda verilen DERS TRANSKRİPTLERİNE bak. Soru burada geçiyorsa doğrudan bu metinlere dayanarak yanıtla ve hangi kaynak dersten yararlandığını parantez içinde veya sonda belirt.
2. Eğer soru transkriptlerde geçmiyorsa ama genetik, biyoinformatik, moleküler ıslah, tarımsal biyoteknoloji veya dijital tarım ile ilgiliyse; yanıtına mutlaka "Bu konu henüz yüklü Genom Okulu derslerinde geçmiyor ama genel bilimsel literatüre göre şöyle açıklayabilirim:" diyerek başla ve kendi bilginle doğru, net bir yanıt ver.
3. Soru tamamen alan dışıysa (spor, siyaset, yemek tarifi, günlük sohbet vb.) sadece biyoinformatik ve Genom Okulu konularında yardımcı olabileceğini söyleyip kibarca reddet.
4. Gerekirse Perl, GMATA veya Python komutlarını kod bloğu içinde temizce göster.

DERS TRANSKRİPTLERİ:
{context}

ÖĞRENCİ SORUSU:
{prompt}
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=system_prompt
            )
            answer = response.text
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})