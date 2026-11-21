import os
import re
import glob
import time
import hashlib
import numpy as np
import streamlit as st
from google import genai
from google.genai import types, errors

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Gene-A(L)İ | Genom Okulu Asistanı",
    page_icon="🧬",
    layout="wide"
)

# --- AYARLAR ---
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 1500          # karakter (büyük parça = daha az embedding isteği)
CHUNK_OVERLAP = 200
# Ücretsiz katman: dakikada 100 embedding. Toplu istekte HER METİN ayrı sayılıyor,
# bu yüzden dakikada en fazla 90 metin gönderiyoruz.
EMBED_PER_MINUTE = 90
CACHE_DIR = ".cache"

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


def get_secret_key():
    """Streamlit Cloud'da secrets'tan okur; yerelde secrets.toml yoksa hata vermez."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    if os.path.exists("logo.jpeg"):
        st.image("logo.jpeg", width="stretch")
    elif os.path.exists("logo.png"):
        st.image("logo.png", width="stretch")

    st.markdown("### 🧬 Gene-A(L)İ v1.1")
    st.markdown(
        """
        **Merhaba! Ben Gene-A(L)İ.** 👋

        **Genom Okulu: 10. Kromozom** platformunun yapay zekâ asistanıyım. 
        Doç. Dr. İbrahim Çelik'in ders içeriklerinden beslenerek; **biyoinformatik, moleküler markör teknolojileri (SSR, SNP, CAPS), genomik analizler ve dijital tarım** konularında sana yardımcı olmak için buradayım.

        *Siyah ekrandan ve kodlardan korkma; sor, birlikte çözelim!*
        """
    )

    api_key = get_secret_key()
    if not api_key:
        st.divider()
        api_key = st.text_input("Gemini API Anahtarı", type="password", placeholder="AIzaSy...")

# Transkriptleri alt klasörler dahil oku
txt_files = sorted(glob.glob("transkriptler/**/*.txt", recursive=True))


# --- METİN PARÇALAMA ---
def chunk_text(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source_name})
        start += chunk_size - overlap
    return chunks


def load_chunks(files):
    all_chunks = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        source = os.path.basename(filepath).replace(".txt", "")
        all_chunks.extend(chunk_text(content, source))
    return all_chunks


def corpus_hash(chunks):
    h = hashlib.sha256()
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for c in chunks:
        h.update(c["source"].encode())
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


def retry_delay_seconds(err, default=60.0):
    """429 hatasındaki 'retry in 16.1s' bilgisini okur."""
    m = re.search(r"retry in ([\d.]+)s", str(err), re.IGNORECASE) or \
        re.search(r"'retryDelay': '(\d+)s'", str(err))
    return float(m.group(1)) + 2 if m else default


def embed_with_retry(client, contents, task_type, max_tries=6):
    for attempt in range(max_tries):
        try:
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in res.embeddings]
        except errors.ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < max_tries - 1:
                time.sleep(retry_delay_seconds(e))
                continue
            raise


def normalize(m):
    m = np.asarray(m, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@st.cache_resource(show_spinner=False)
def build_vector_store(_api_key, files):
    chunks = load_chunks(files)
    if not chunks:
        return None, []

    # Disk önbelleği: aynı transkriptler için embedding'leri bir daha hesaplama
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"embeddings_{corpus_hash(chunks)}.npy")
    if os.path.exists(cache_file):
        return np.load(cache_file), chunks

    client = genai.Client(api_key=_api_key)
    texts = [c["text"] for c in chunks]
    embeddings = []

    total_batches = (len(texts) + EMBED_PER_MINUTE - 1) // EMBED_PER_MINUTE
    bar = st.progress(0.0, text="Gene-A(L)İ ders notlarını ilk kez hazırlıyor...")
    for b, i in enumerate(range(0, len(texts), EMBED_PER_MINUTE)):
        t0 = time.time()
        embeddings.extend(
            embed_with_retry(client, texts[i:i + EMBED_PER_MINUTE], "RETRIEVAL_DOCUMENT")
        )
        bar.progress(
            (b + 1) / total_batches,
            text=f"Ders notları hazırlanıyor: {len(embeddings)}/{len(texts)} parça "
                 f"(ücretsiz kota nedeniyle dakikada {EMBED_PER_MINUTE} parça)",
        )
        # Dakikalık kotayı aşmamak için bir sonraki pakete kadar bekle
        if i + EMBED_PER_MINUTE < len(texts):
            time.sleep(max(0.0, 61 - (time.time() - t0)))
    bar.empty()

    matrix = normalize(embeddings)
    np.save(cache_file, matrix)
    return matrix, chunks


def search(matrix, chunks, query_vec, k=4):
    scores = matrix @ query_vec
    top = np.argsort(-scores)[:min(k, len(chunks))]
    return [chunks[i] for i in top]


# --- ANA EKRAN ---
st.title(">_ Gene-A(L)İ | Genom Okulu")
st.caption("Biyoinformatik, Moleküler Markör Tasarımı ve Dijital Tarım Asistanı")

if not api_key:
    st.info("Sistem şu an bakım/kurulum aşamasında. (Yönetici notu: GEMINI_API_KEY tanımlanmalıdır.)")
    st.stop()

if not txt_files:
    st.warning("Ders kütüphanesi henüz yüklenmedi. Lütfen daha sonra tekrar deneyin.")
    st.stop()

try:
    matrix, chunks = build_vector_store(api_key, tuple(txt_files))
except errors.APIError as e:
    st.error("Ders notları hazırlanırken Gemini API hatası oluştu. Birkaç dakika sonra sayfayı yenileyin.")
    st.caption(f"Ayrıntı: {getattr(e, 'code', '')} {getattr(e, 'status', '')}")
    st.stop()

if matrix is None:
    st.warning("Ders kütüphanesi boş görünüyor.")
    st.stop()

client = genai.Client(api_key=api_key)

# Sohbet geçmişi ve karşılama mesajı
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Merhaba! Ben **Gene-A(L)İ**, Genom Okulu'nun yapay zekâ asistanıyım. 🧬\n\n"
                "Genom Okulu derslerinde işlenen **GMATA ile SSR tasarımı, FASTA dosyaları, moleküler markörler, biyoinformatik analizler ve dijital tarım** "
                "hakkında aklına takılan her şeyi bana sorabilirsin. Nereden başlayalım?"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

SYSTEM_PROMPT = """Sen 'Gene-A(L)İ'sin; Doç. Dr. İbrahim Çelik'in kurduğu 'Genom Okulu: 10. Kromozom' platformunun yapay zekâ asistanısın.
Görevin: Öğrencilerin sorularını "Halk Ekmek" sadeliğinde ama akademik derinlikte, samimi bir dille yanıtlamak.

Kurallar:
1. Öncelikle verilen DERS TRANSKRİPTLERİNE bak. Soru burada geçiyorsa doğrudan bu metinlere dayanarak yanıtla ve hangi kaynak dersten yararlandığını parantez içinde veya sonda belirt.
2. Transkriptler YouTube otomatik altyazısıdır; terimler yanlış yazılmış olabilir (ör. "Sasar"=SSR, "Civas"=GWAS, "EFP"=AFLP, "snip"=SNP). Yanıtında doğru terimleri kullan.
3. Eğer soru transkriptlerde geçmiyorsa ama genetik, biyoinformatik, moleküler ıslah, tarımsal biyoteknoloji veya dijital tarım ile ilgiliyse; yanıtına mutlaka "Bu konu henüz yüklü Genom Okulu derslerinde geçmiyor ama genel bilimsel literatüre göre şöyle açıklayabilirim:" diyerek başla ve kendi bilginle doğru, net bir yanıt ver.
4. Soru tamamen alan dışıysa (spor, siyaset, yemek tarifi, günlük sohbet vb.) sadece biyoinformatik ve Genom Okulu konularında yardımcı olabileceğini söyleyip kibarca reddet.
5. Gerekirse Perl, GMATA veya Python komutlarını kod bloğu içinde temizce göster.
"""

# Kullanıcı sorusu
if prompt := st.chat_input("Örn: GMATA'da mononükleotitleri neden eliyoruz?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Genom taranıyor..."):
                q_vec = normalize(embed_with_retry(client, [prompt], "RETRIEVAL_QUERY", max_tries=3))[0]
                retrieved = search(matrix, chunks, q_vec, k=4)
                context = "\n\n".join(
                    [f"[Kaynak Ders: {c['source']}]\n{c['text']}" for c in retrieved]
                )
                response = client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=f"DERS TRANSKRİPTLERİ:\n{context}\n\nÖĞRENCİ SORUSU:\n{prompt}",
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                answer = response.text or "Bu soruya şu an yanıt üretemedim, lütfen farklı bir şekilde sormayı dene."
        except errors.APIError as e:
            if getattr(e, "code", None) == 429:
                answer = "⏳ Şu an çok yoğunum (ücretsiz kota doldu). Lütfen bir dakika sonra tekrar sor."
            else:
                answer = f"⚠️ Yapay zekâ servisine ulaşırken bir sorun oluştu ({getattr(e, 'code', '')}). Lütfen tekrar dene."
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
