# Streamlit Community Cloud ships an old sqlite3 that Chroma rejects.
# Swap in the newer one from pysqlite3-binary when it's available.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="EDM Music Search", page_icon="🎛️", layout="centered")


# Load the model and database once, then reuse them across searches
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chromadb")
    return client.get_collection("edm_songs")


model = load_model()
collection = load_collection()


@st.cache_data
def get_genres():
    metas = collection.get(include=["metadatas"])["metadatas"]
    return ["Any"] + sorted({m["track_genre"] for m in metas})


# ---------- Page layout ----------
st.title("🎛️ EDM Music Search")
st.write("Describe the vibe you want and find matching tracks.")

EXAMPLES = ["chill track to study to", "EDM festival sets",
            "summer house", "instrumental techno"]

if "query" not in st.session_state:
    st.session_state.query = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.query = ex

col1, col2 = st.columns([3, 1])
query = col1.text_input("Describe a vibe", key="query",
                        placeholder="techno, pop remix, dance")
genre = col2.selectbox("Genre", get_genres())
n_results = st.slider("Number of results", 1, 10, 5)

# ---------- Search ----------
if query.strip():
    where = None if genre == "Any" else {"track_genre": genre}
    results = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n_results,
        where=where,
    )
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not metas:
        st.warning("No matches. Try another genre.")

    for meta, dist in zip(metas, dists):
        match = round((1 - dist) * 100)
        link = f"https://open.spotify.com/track/{meta['track_id']}"
        with st.container(border=True):
            st.subheader(meta["track_name"])
            st.markdown(
                f"**{meta['artists']}** · {meta['track_genre']} · "
                f"{round(meta['tempo'])} BPM · {match}% match · "
                f"[Listen on Spotify]({link})"
            )
