import os
import io
import re
import glob
import time
import base64
import random
import hashlib
import numpy as np
import streamlit as st
from google import genai
from google.genai import types, errors

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Gene-A(L)İ | Genom Okulu Asistanı",
    page_icon="🧬",
    layout="centered"
)

# --- AYARLAR ---
EMBED_MODEL = "gemini-embedding-001"
# Sırayla denenir; Google bir modeli kapatınca (404) bir sonrakine geçilir.
# "gemini-flash-latest" her zaman güncel Flash modeline yönlendirilen takma addır.
CHAT_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.7-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"]
ANSWER_TIME_BUDGET = 50  # sn: bu süre dolunca yeni model denenmez, öğrenciye "yoğunuz" denir
CHUNK_SIZE = 1500          # karakter (büyük parça = daha az embedding isteği)
CHUNK_OVERLAP = 200
# Ücretsiz katman: dakikada 100 istek VE dakikada sınırlı token. Toplu istekte HER METİN
# ayrı sayılıyor; 1500 karakterlik 30 parça ~15 bin token eder, iki sınırın da altında kalır.
EMBED_PER_MINUTE = 30
HISTORY_TURNS = 6          # modele gönderilen son mesaj sayısı (takip soruları için)
CACHE_DIR = ".cache"
LIBRARY_DIR = "transkriptler"   # .txt ve .pdf dosyaları (alt klasörler dahil)

YOUTUBE_URL = "https://www.youtube.com/@genomokulu"
WEB_URL = "https://www.pau.edu.tr/icelik/tr"

THINKING_MESSAGES = [
    "Ders notları karıştırılıyor...",
    "Kromozomlar taranıyor...",
    "SSR motifleri sayılıyor...",
    "Primerler hizalanıyor...",
    "Restriksiyon enzimleri kesime hazırlanıyor...",
    "Aleller karşılaştırılıyor...",
    "Jel yürütülüyor, bantlar bekleniyor...",
    "SNP'ler ayıklanıyor...",
    "PCR döngüleri çalışıyor...",
    "FASTA dosyaları okunuyor...",
]

# --- TEMA (CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;600&display=swap');

.stApp { background: radial-gradient(ellipse at top, #10201a 0%, #0d1117 55%) fixed; color: #c9d1d9; }
.block-container { padding-top: 1.2rem !important; max-width: 860px; }
header[data-testid="stHeader"] { background: transparent; }

/* --- Giriş (hero) --- */
.hero { text-align: center; margin: 0 auto 0.6rem auto; }
.hero img.logo {
    width: 170px; height: 170px; border-radius: 50%;
    box-shadow: 0 0 18px rgba(57,255,20,.45), 0 0 46px rgba(0,229,255,.25);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 18px rgba(57,255,20,.45), 0 0 46px rgba(0,229,255,.25); }
    50%     { box-shadow: 0 0 26px rgba(255,46,136,.45), 0 0 60px rgba(57,255,20,.30); }
}
.hero .title {
    font-family: 'Orbitron', sans-serif; font-weight: 900; letter-spacing: 3px;
    font-size: 3.1rem; line-height: 1.1; margin: .7rem 0 .2rem 0;
}
.hero .g, .side-title .g { color: #39ff14; text-shadow: 0 0 8px rgba(57,255,20,.8), 0 0 22px rgba(57,255,20,.4); }
.hero .p, .side-title .p { color: #ff2e88; text-shadow: 0 0 8px rgba(255,46,136,.9), 0 0 22px rgba(255,46,136,.5); }
.hero .y, .side-title .y { color: #e8ff1a; text-shadow: 0 0 8px rgba(232,255,26,.9), 0 0 22px rgba(232,255,26,.4); }
.hero .sub {
    font-family: 'Orbitron', sans-serif; font-weight: 700; letter-spacing: 4px; font-size: .95rem;
}
.hero .sub .c { color: #00e5ff; text-shadow: 0 0 8px rgba(0,229,255,.7); }
.hero .tag { color: #8b949e; font-family: 'Inter', sans-serif; font-size: .92rem; margin-top: .35rem; }
.ai-note {
    font-family: 'Inter', sans-serif; font-size: .78rem; color: #d29922;
    border: 1px solid rgba(210,153,34,.35); background: rgba(210,153,34,.07);
    border-radius: 8px; padding: 6px 12px; margin: .6rem auto 0 auto; display: inline-block;
}

/* --- Sohbet --- */
.stChatMessage {
    background-color: rgba(22,27,34,.85); border: 1px solid #30363d; border-radius: 12px; padding: 10px;
}
.stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage td { color: #e6edf3 !important; }
.stChatMessage code { color: #39ff14 !important; }

/* Soru kutusu: koyu zemin, açık yazı */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: #0d1117 !important; }
[data-testid="stChatInput"] { background: #161b22 !important; border: 1px solid #39ff14 !important; border-radius: 12px !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] > div > div { background: #161b22 !important; }
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important; -webkit-text-fill-color: #e6edf3 !important;
    background: transparent !important; caret-color: #39ff14 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #6e7681 !important; -webkit-text-fill-color: #6e7681 !important; }

/* Yazılara dönen spinner */
[data-testid="stSpinner"] * { color: #39ff14 !important; }

/* --- Yan menü --- */
[data-testid="stSidebar"] { background-color: #010409; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9; }
.side-title { font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 1.35rem; letter-spacing: 2px; }
.side-links a {
    display: block; text-decoration: none; margin: 6px 0; padding: 8px 12px; border-radius: 8px;
    border: 1px solid #30363d; background: #0d1117; color: #e6edf3 !important; font-weight: 600;
}
.side-links a:hover { border-color: #39ff14; box-shadow: 0 0 10px rgba(57,255,20,.35); }
[data-testid="stSidebar"] button { background: #0d1117 !important; border: 1px solid #30363d !important; }
[data-testid="stSidebar"] button * { color: #e6edf3 !important; }
[data-testid="stSidebar"] button:hover { border-color: #39ff14 !important; box-shadow: 0 0 10px rgba(57,255,20,.35); }
</style>
""", unsafe_allow_html=True)


# --- YARDIMCI: LOGO ---
@st.cache_data(show_spinner=False)
def logo_data_uri(path, size=340):
    """Logoyu küçültüp sayfaya gömülebilir hale getirir."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data = buf.getvalue()
    except Exception:
        with open(path, "rb") as f:
            data = f.read()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def find_logo():
    for p in ("logo.jpeg", "logo.jpg", "logo.png"):
        if os.path.exists(p):
            return p
    return None


def get_secret_key():
    """Streamlit Cloud'da secrets'tan okur; yerelde secrets.toml yoksa hata vermez."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


TITLE_HTML = ('<span class="g">GENE-</span><span class="p">A</span>'
              '<span class="y">(L)</span><span class="p">İ</span>')

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    st.markdown(f'<div class="side-title">{TITLE_HTML}</div>', unsafe_allow_html=True)
    st.caption("Genom Okulu: 10. Kromozom'un yapay zekâ asistanı")
    st.markdown(
        """
**Ne yapar?**
Doç. Dr. İbrahim Çelik'in Genom Okulu derslerini ve yayınlarını okuyarak sorularını yanıtlar;
yanıtın hangi dersten geldiğini de söyler.

**Neler sorabilirsin?**
- SSR, SNP ve CAPS markörü tasarımı
- Primer tasarımı ve PCR / jel yorumlama
- GMATA, dbSNP2CAPS gibi araçların kullanımı
- Genetik çeşitlilik, haritalama, GWAS
- Biyoinformatik veri tabanları (NCBI, SGN...)
        """
    )
    st.markdown(
        f"""
<div class="side-links">
  <a href="{YOUTUBE_URL}" target="_blank">▶️ YouTube: Genom Okulu</a>
  <a href="{WEB_URL}" target="_blank">🌐 Web: Doç. Dr. İbrahim Çelik</a>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("⚠️ Gene-A(L)İ bir yapay zekâdır ve hata yapabilir. Bilgiler YouTube'un otomatik "
               "altyazılarından derlendiği için terimler yer yer yanlış olabilir; önemli konuları "
               "ders videolarından ve kaynak yayınlardan doğrulayın.")

    api_key = get_secret_key()
    if not api_key:
        api_key = st.text_input("Gemini API Anahtarı", type="password", placeholder="AIzaSy...")
    if st.button("🧹 Sohbeti temizle"):
        st.session_state.pop("messages", None)
        st.rerun()

# --- GİRİŞ (HERO) ---
_logo = find_logo()
_logo_html = f'<img class="logo" src="{logo_data_uri(_logo)}"/>' if _logo else ""
st.markdown(
    f"""
<div class="hero">
  {_logo_html}
  <div class="title">{TITLE_HTML}</div>
  <div class="sub"><span class="g">GENOM OKULU</span> · <span class="c">10. KROMOZOM</span></div>
  <div class="tag">Biyoinformatik ve Moleküler Markör Asistanı</div>
  <div class="ai-note">⚠️ Yapay zekâ yanıtları hatalı olabilir; önemli bilgileri ders videolarından doğrulayın.</div>
</div>
    """,
    unsafe_allow_html=True,
)


# --- KÜTÜPHANE: TXT + PDF ---
def library_files():
    files = glob.glob(f"{LIBRARY_DIR}/**/*.txt", recursive=True) + \
            glob.glob(f"{LIBRARY_DIR}/**/*.pdf", recursive=True)
    return sorted(files)


def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    # Satır sonu tirelerini ve fazla boşlukları temizle
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


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
        try:
            if filepath.lower().endswith(".pdf"):
                content = read_pdf(filepath)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as e:
            print(f"[Gene-AL] okunamadı: {filepath}: {e}", flush=True)
            continue
        source = os.path.splitext(os.path.basename(filepath))[0]
        kind = "pdf" if filepath.lower().endswith(".pdf") else "txt"
        for c in chunk_text(content, source):
            c["kind"] = kind
            all_chunks.append(c)
    return all_chunks


def chunk_key(c):
    return hashlib.sha256(f"{EMBED_MODEL}|{c['source']}|{c['text']}".encode()).hexdigest()[:20]


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
        except errors.ServerError as e:
            # 500/503: Google tarafı geçici olarak yoğun; kısa bekleyip tekrar dene
            print(f"[Gene-AL] embedding sunucu hatası {getattr(e, 'code', '')} (deneme {attempt + 1}/{max_tries})", flush=True)
            if attempt == max_tries - 1:
                raise
            time.sleep(2 + attempt * 2)
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


STORE_FILE = os.path.join(CACHE_DIR, "embedding_store.npz")
# Depoya konan hazır vektörler: sunucu uykudan uyanınca bile silinmez,
# böylece ders notları baştan işlenmez. (?admin=1 ile yan menüden indirilebilir.)
SEED_FILE = "vektor_onbellek.npz"


def _read_npz(path):
    try:
        d = np.load(path, allow_pickle=False)
        return dict(zip(d["keys"].tolist(), d["vecs"]))
    except Exception as e:
        print(f"[Gene-AL] önbellek okunamadı: {path}: {e}", flush=True)
        return {}


def load_store():
    store = {}
    if os.path.exists(SEED_FILE):
        store.update(_read_npz(SEED_FILE))
    if os.path.exists(STORE_FILE):
        store.update(_read_npz(STORE_FILE))
    return store


def export_store_bytes(chunks, matrix):
    """Şu anki kütüphanenin vektörlerini depoya yüklenecek tek dosya olarak verir."""
    buf = io.BytesIO()
    np.savez_compressed(buf, keys=np.asarray([chunk_key(c) for c in chunks]),
                        vecs=np.asarray(matrix, dtype="float32"))
    return buf.getvalue()


def save_store(store):
    os.makedirs(CACHE_DIR, exist_ok=True)
    keys = list(store.keys())
    vecs = np.asarray([store[k] for k in keys], dtype="float32")
    tmp = STORE_FILE + ".tmp.npz"
    np.savez(tmp, keys=np.asarray(keys), vecs=vecs)
    os.replace(tmp, STORE_FILE)


def migrate_old_cache(chunks, keys, store):
    """v1.1'in tek dosyalık önbelleğini (embeddings_<hash>.npy) yeni parça bazlı yapıya aktarır."""
    txt = [(c, k) for c, k in zip(chunks, keys) if c.get("kind") == "txt"]
    h = hashlib.sha256()
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for c, _ in txt:
        h.update(c["source"].encode())
        h.update(c["text"].encode())
    old = os.path.join(CACHE_DIR, f"embeddings_{h.hexdigest()[:16]}.npy")
    if os.path.exists(old) and any(k not in store for _, k in txt):
        try:
            m = np.load(old)
            if len(m) == len(txt):
                for (_, k), v in zip(txt, m):
                    store.setdefault(k, v)
                save_store(store)
                print(f"[Gene-AL] eski önbellekten {len(txt)} parça aktarıldı", flush=True)
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def build_vector_store(_api_key, files):
    """Her parçanın vektörü ayrı saklanır: yeni bir PDF eklenince yalnızca yeni parçalar işlenir."""
    chunks = load_chunks(files)
    if not chunks:
        return None, []

    store = load_store()
    keys = [chunk_key(c) for c in chunks]
    migrate_old_cache(chunks, keys, store)
    missing = [i for i, k in enumerate(keys) if k not in store]

    if missing:
        client = genai.Client(api_key=_api_key)
        done = len(chunks) - len(missing)
        bar = st.progress(done / len(chunks), text="Gene-A(L)İ yeni ders notlarını hazırlıyor...")

        def status(msg):
            bar.progress(done / len(chunks), text=msg)

        for j in range(0, len(missing), EMBED_PER_MINUTE):
            t0 = time.time()
            idx = missing[j:j + EMBED_PER_MINUTE]
            vecs = embed_with_retry(client, [chunks[i]["text"] for i in idx], "RETRIEVAL_DOCUMENT", status=status)
            for i, v in zip(idx, vecs):
                store[keys[i]] = np.asarray(v, dtype="float32")
            save_store(store)          # yarıda kesilirse kaldığı yerden devam eder
            done += len(idx)
            print(f"[Gene-AL] embedding: {done}/{len(chunks)}", flush=True)
            status(f"Ders notları hazırlanıyor: {done}/{len(chunks)} parça "
                   f"(ücretsiz kota nedeniyle dakikada {EMBED_PER_MINUTE} parça, yalnızca yeni dosyalar için)")
            if j + EMBED_PER_MINUTE < len(missing):
                time.sleep(max(0.0, 61 - (time.time() - t0)))
        bar.empty()

    matrix = normalize([store[k] for k in keys])
    return matrix, chunks


SYSTEM_PROMPT = """Sen 'Gene-A(L)İ'sin; Doç. Dr. İbrahim Çelik'in kurduğu 'Genom Okulu: 10. Kromozom' platformunun yapay zekâ asistanısın.
Görevin: Öğrencilerin sorularını sade ama akademik derinlikte, samimi bir dille yanıtlamak.

Kurallar:
1. Öncelikle verilen KAYNAKLARA (ders transkriptleri ve İbrahim Çelik'in yayınları) bak. Soru burada geçiyorsa bu metinlere dayanarak yanıtla ve yanıtın sonunda hangi kaynaktan yararlandığını belirt.
2. Transkriptler YouTube otomatik altyazısıdır; terimler yanlış yazılmış olabilir (ör. "Sasar"=SSR, "Civas"=GWAS, "EFP"=AFLP, "snip"=SNP). Yanıtında doğru terimleri kullan. Altyazıda belirsiz kalan sayı ve isimleri kesin bilgi gibi sunma.
3. Soru kaynaklarda geçmiyorsa ama genetik, biyoinformatik, moleküler ıslah veya tarımsal biyoteknoloji ile ilgiliyse, yanıtına "Bu konu henüz yüklü Genom Okulu derslerinde geçmiyor ama genel bilimsel literatüre göre şöyle açıklayabilirim:" diyerek başla ve kendi bilginle doğru, net bir yanıt ver.
4. Soru tamamen alan dışıysa (spor, siyaset, yemek tarifi vb.) sadece Genom Okulu konularında yardımcı olabileceğini söyleyip kibarca reddet.
5. ÖNCEKİ KONUŞMA'yı dikkate al; takip sorularını önceki mesajlarla birlikte anla.
6. Her yanıta selamlaşmayla başlama ve kalıp ifadeleri ("Halk Ekmek sadeliğinde" gibi) tekrarlama; doğrudan konuya gir.
7. Gerekirse Perl, GMATA veya Python komutlarını kod bloğu içinde temizce göster.
"""


def generate_answer(client, contents):
    """Çalışan ilk sohbet modelini bulur ve oturum boyunca onu kullanır."""
    models = CHAT_MODELS
    if st.session_state.get("chat_model"):
        models = [st.session_state["chat_model"]] + [m for m in CHAT_MODELS if m != st.session_state["chat_model"]]
    last_err = None
    started = time.time()
    for model in models:
        if last_err is not None and time.time() - started > ANSWER_TIME_BUDGET:
            print("[Gene-AL] süre doldu, yeni model denenmiyor", flush=True)
            break
        for attempt in range(1):  # başarısız denemeler de kotadan yer; her model bir kez
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )
                st.session_state["chat_model"] = model
                return response.text
            except errors.ServerError as e:
                # 500/503/504 = Google sunucusu yoğun: sonraki modele geç
                print(f"[Gene-AL] {model} sunucu hatası {getattr(e, 'code', '')}, sonraki modele geçiliyor", flush=True)
                last_err = e
                break
            except errors.ClientError as e:
                code = getattr(e, "code", None)
                if code in (404, 429):
                    # 404: model kapandı; 429: bu modelin kotası doldu -> sıradaki model
                    print(f"[Gene-AL] {model} kullanılamıyor ({code}), sonraki modele geçiliyor", flush=True)
                    last_err = e
                    break
                raise
            except Exception as e:
                # Zaman aşımı / bağlantı kopması -> sıradaki model
                if "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower() or "connect" in type(e).__name__.lower():
                    print(f"[Gene-AL] {model} zaman aşımı/bağlantı: {type(e).__name__}", flush=True)
                    last_err = e
                    break
                raise
    raise last_err


def search(matrix, chunks, query_vec, k=5):
    scores = matrix @ query_vec
    top = np.argsort(-scores)[:min(k, len(chunks))]
    return [chunks[i] for i in top]


def history_text(messages):
    """Karşılama mesajı hariç son mesajları düz metne çevirir."""
    recent = [m for m in messages[1:]][-HISTORY_TURNS:]
    lines = []
    for m in recent:
        who = "Öğrenci" if m["role"] == "user" else "Gene-A(L)İ"
        lines.append(f"{who}: {m['content'][:1200]}")
    return "\n".join(lines)


# --- ANA AKIŞ ---
if not api_key:
    st.info("Sistem şu an bakım/kurulum aşamasında. (Yönetici notu: GEMINI_API_KEY tanımlanmalıdır.)")
    st.stop()

files = library_files()
if not files:
    st.warning("Ders kütüphanesi henüz yüklenmedi. Lütfen daha sonra tekrar deneyin.")
    st.stop()

try:
    matrix, chunks = build_vector_store(api_key, tuple(files))
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

def make_client(key):
    """Takılıp kalmasın: her istek en fazla 25 sn beklesin, kütüphanenin uzun iç yeniden denemeleri kapalı."""
    try:
        return genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=25000, retry_options=types.HttpRetryOptions(attempts=1)))
    except Exception:
        try:
            return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=25000))
        except Exception:
            return genai.Client(api_key=key)


client = make_client(api_key)

# Yönetici: hazır vektörleri indir (adresin sonuna ?admin=1 ekleyince görünür)
if st.query_params.get("admin") == "1":
    with st.sidebar:
        st.divider()
        st.markdown("**🔧 Yönetici**")
        st.caption(f"Kütüphane: {len(files)} dosya, {len(chunks)} parça. Bu dosyayı GitHub deposunun "
                   f"ana klasörüne **{SEED_FILE}** adıyla yükleyin; uygulama uykudan uyanınca "
                   f"ders notlarını baştan işlemez.")
        st.download_button("⬇️ Vektörleri indir", data=export_store_bytes(chunks, matrix),
                           file_name=SEED_FILE, mime="application/octet-stream")

# Sohbet geçmişi ve karşılama mesajı
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Merhaba! Ben **Gene-A(L)İ**, Genom Okulu'nun yapay zekâ asistanıyım. 🧬\n\n"
                "**SSR, SNP ve CAPS markörleri, primer tasarımı, GMATA, FASTA dosyaları ve biyoinformatik analizler** "
                "hakkında aklına takılan her şeyi sorabilirsin. Nereden başlayalım?"
            )
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcı sorusu
if prompt := st.chat_input("Sorunu yaz... (ör. GMATA'da mononükleotitleri neden eliyoruz?)"):
    past = history_text(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner(random.choice(THINKING_MESSAGES)):
                # Takip sorularında önceki soruyu da aramaya kat
                prev_q = next((m["content"] for m in reversed(st.session_state.messages[:-1])
                               if m["role"] == "user"), "")
                search_text = f"{prev_q}\n{prompt}" if prev_q else prompt
                q_vec = normalize(embed_with_retry(client, [search_text], "RETRIEVAL_QUERY", max_tries=3))[0]
                retrieved = search(matrix, chunks, q_vec, k=5)
                context = "\n\n".join(
                    [f"[Kaynak: {c['source']}]\n{c['text']}" for c in retrieved]
                )
                text = generate_answer(
                    client,
                    f"KAYNAKLAR:\n{context}\n\n"
                    f"ÖNCEKİ KONUŞMA:\n{past or '(yok)'}\n\n"
                    f"ÖĞRENCİ SORUSU:\n{prompt}"
                )
                answer = text or "Bu soruya şu an yanıt üretemedim, lütfen farklı bir şekilde sormayı dene."
        except DailyQuotaError:
            answer = "⏳ Bugünlük ücretsiz kotam doldu. Yarın sabah tekrar sorabilirsin."
        except errors.APIError as e:
            if getattr(e, "code", None) == 429:
                answer = "⏳ Şu an çok yoğunum (ücretsiz kota doldu). Lütfen bir dakika sonra tekrar sor."
            elif getattr(e, "code", None) in (500, 502, 503, 504):
                answer = "🧬 Şu an Google'ın yapay zekâ sunucuları çok yoğun, birkaç saniye sonra tekrar sorar mısın?"
            else:
                answer = f"⚠️ Yapay zekâ servisine ulaşırken bir sorun oluştu ({getattr(e, 'code', '')}). Lütfen tekrar dene."
        except Exception as e:
            print(f"[Gene-AL] beklenmeyen hata: {type(e).__name__}: {e}", flush=True)
            answer = "🧬 Şu an Google'ın yapay zekâ sunucuları çok yoğun, birkaç saniye sonra tekrar sorar mısın?"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
