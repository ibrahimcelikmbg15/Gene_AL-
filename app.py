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
# Sırayla denenir; Google bir modeli kapatınca (404) bir sonrakine geçilir.
# "gemini-flash-latest" her zaman güncel Flash modeline yönlendirilen takma addır.
CHAT_MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-2.5-flash"]
CHUNK_SIZE = 1500          # karakter (büyük parça = daha az embedding isteği)
CHUNK_OVERLAP = 200
# Ücretsiz katman: dakikada 100 istek VE dakikada sınırlı token. Toplu istekte HER METİN
# ayrı sayılıyor; 1500 karakterlik 30 parça ~15 bin token eder, iki sınırın da altında kalır.
EMBED_PER_MINUTE = 30
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
    .stChatMessage p, .stChatMessage li, .stChatMessage span {
        color: #e6edf3 !important;
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


class DailyQuotaError(Exception):
    pass


def quota_ids(err):
    return re.findall(r"'quotaId': '([^']+)'", str(err))


def embed_with_retry(client, contents, task_type, max_tries=6, status=None):
    for attempt in range(max_tries):
        try:
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in res.embeddings]
        except errors.ClientError as e:
            if getattr(e, "code", None) != 429:
                raise
            ids = quota_ids(e)
            print(f"[Gene-AL] 429 kota hatası (deneme {attempt + 1}/{max_tries}): {ids}", flush=True)
            if any("PerDay" in q for q in ids):
                raise DailyQuotaError(", ".join(ids)) from e
            if attempt == max_tries - 1:
                raise
            wait = retry_delay_seconds(e)
            if status is not None:
                status(f"Gemini kotası doldu, {int(wait)} sn bekleniyor... ({', '.join(ids) or 'dakikalık sınır'})")
            time.sleep(wait)


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

    # Yarım kalan iş varsa kaldığı yerden devam et
    partial_file = cache_file.replace(".npy", "_partial.npy")
    embeddings = list(np.load(partial_file)) if os.path.exists(partial_file) else []

    bar = st.progress(len(embeddings) / len(texts), text="Gene-A(L)İ ders notlarını ilk kez hazırlıyor...")

    def status(msg):
        bar.progress(len(embeddings) / len(texts), text=msg)

    while len(embeddings) < len(texts):
        i = len(embeddings)
        t0 = time.time()
        embeddings.extend(
            embed_with_retry(client, texts[i:i + EMBED_PER_MINUTE], "RETRIEVAL_DOCUMENT", status=status)
        )
        np.save(partial_file, np.asarray(embeddings, dtype="float32"))
        print(f"[Gene-AL] embedding: {len(embeddings)}/{len(texts)}", flush=True)
        status(f"Ders notları hazırlanıyor: {len(embeddings)}/{len(texts)} parça "
               f"(ücretsiz kota nedeniyle dakikada {EMBED_PER_MINUTE} parça, yalnızca ilk açılışta)")
        # Dakikalık kotayı aşmamak için bir sonraki pakete kadar bekle
        if len(embeddings) < len(texts):
            time.sleep(max(0.0, 61 - (time.time() - t0)))
    bar.empty()

    matrix = normalize(embeddings)
    np.save(cache_file, matrix)
    if os.path.exists(partial_file):
        os.remove(partial_file)
    return matrix, chunks


def generate_answer(client, contents):
    """Çalışan ilk sohbet modelini bulur ve oturum boyunca onu kullanır."""
    models = CHAT_MODELS
    if st.session_state.get("chat_model"):
        models = [st.session_state["chat_model"]] + [m for m in CHAT_MODELS if m != st.session_state["chat_model"]]
    last_err = None
    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            st.session_state["chat_model"] = model
            return response.text
        except errors.ClientError as e:
            if getattr(e, "code", None) == 404:
                print(f"[Gene-AL] model bulunamadı (404): {model}", flush=True)
                last_err = e
                continue
            raise
    raise last_err


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
except DailyQuotaError:
    st.error("Gemini API'nin günlük ücretsiz kotası bugünlük doldu. Kota her gün Türkiye saatiyle "
             "yaklaşık 10:00'da sıfırlanır; o zaman sayfayı yenileyin. (Hazırlanan kısım kaydedildi.)")
    st.stop()
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
                text = generate_answer(
                    client, f"DERS TRANSKRİPTLERİ:\n{context}\n\nÖĞRENCİ SORUSU:\n{prompt}"
                )
                answer = text or "Bu soruya şu an yanıt üretemedim, lütfen farklı bir şekilde sormayı dene."
        except DailyQuotaError:
            answer = "⏳ Bugünlük ücretsiz kotam doldu. Yarın sabah tekrar sorabilirsin."
        except errors.APIError as e:
            if getattr(e, "code", None) == 429:
                answer = "⏳ Şu an çok yoğunum (ücretsiz kota doldu). Lütfen bir dakika sonra tekrar sor."
            else:
                answer = f"⚠️ Yapay zekâ servisine ulaşırken bir sorun oluştu ({getattr(e, 'code', '')}). Lütfen tekrar dene."
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
