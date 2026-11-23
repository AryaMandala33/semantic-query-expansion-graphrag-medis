"""
app.py — Antarmuka Streamlit sistem utama (Bab III 3.3).

Alur: Query Service -> LLM Expander -> (Clarification Module untuk specificity gap)
      -> Disambiguation Ranker -> hasil retrieval -> visualisasi sub-graf.

Jalankan dari folder Project:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Disambiguasi Istilah Medis — GraphRAG", page_icon="🧬",
                   layout="wide")

TYPE_LABELS = {"acronym": "Akronim medis", "polysemy": "Polisemi medis",
               "specificity_gap": "Specificity gap", "none": "Tidak terdeteksi"}
STATUS_LABELS = {"mapped": "terpetakan", "type_mismatch": "tidak terpetakan (tipe node berbeda)",
                 "not_found": "tidak ditemukan di graf"}
LABEL_COLORS = {"Disease": "#d1495b", "Phenotype": "#edae49", "Drug": "#00798c",
                "Protein": "#7a5195"}


@st.cache_resource(show_spinner="Memuat model, LLM, dan koneksi Neo4j...")
def load_systems():
    from core.baseline import BaselineSystem
    from core.pipeline import MainSystem
    ms = MainSystem()
    return ms, BaselineSystem(retriever=ms.retriever, ranker=ms.ranker)


def subgraph_figure(interp, ranked_nodes, max_nodes=15):
    import networkx as nx
    import plotly.graph_objects as go

    concept = interp["node"]
    G = nx.Graph()
    G.add_node(concept["node_index"], name=concept["name"], label=concept["label"], center=True)
    shown = [n for n in ranked_nodes if n["node_index"] != concept["node_index"]][:max_nodes]
    for n in shown:
        G.add_node(n["node_index"], name=n["name"], label=n["label"])
        G.add_edge(concept["node_index"], n["node_index"],
                   relation=n.get("display_relation") or n.get("relation") or "")
    pos = nx.spring_layout(G, seed=42, k=0.9)

    fig = go.Figure()
    ex, ey, mx, my, mt = [], [], [], [], []
    for a, b, data in G.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        mt.append(data["relation"])
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", showlegend=False, hoverinfo="skip",
                             line=dict(width=1, color="#9aa0a6")))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", showlegend=False, text=mt,
                             hoverinfo="text", marker=dict(size=6, opacity=0)))
    for label, color in LABEL_COLORS.items():
        ids = [i for i, d in G.nodes(data=True) if d["label"] == label]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids], mode="markers+text",
            name=label, text=[G.nodes[i]["name"] for i in ids], textposition="top center",
            hoverinfo="text",
            marker=dict(size=[30 if G.nodes[i].get("center") else 15 for i in ids], color=color,
                        line=dict(width=1, color="#222"))))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.05))
    return fig


# =====================================================================
st.title("🧬 Disambiguasi Istilah Medis pada GraphRAG")
st.caption("Semantic Query Expansion berbasis LLM pada subset PrimeKG. "
           "Prototipe penelitian — bukan alat diagnosis atau rekomendasi pengobatan.")

for key, default in [("state", None), ("result", None)]:
    st.session_state.setdefault(key, default)

with st.form("input"):
    query = st.text_input("Kueri medis (bahasa Indonesia, Inggris, atau campuran)",
                          placeholder="Contoh: obat untuk MS pada pasien dengan neuritis optik")
    submitted = st.form_submit_button("Proses kueri", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Masukkan kueri terlebih dahulu.")
    else:
        ms, _ = load_systems()
        with st.spinner("Query Service, LLM Expander, dan pemetaan ke graf..."):
            state = ms.process(query.strip())
        st.session_state.state = state
        st.session_state.result = (None if state.get("needs_clarification") or state.get("error")
                                   else ms.finalize(state))

state = st.session_state.state
if state is None:
    st.stop()
if state.get("error"):
    st.error(state["error"])
    st.stop()

ms, base = load_systems()
qs, exp = state["query_service"], state["expansion"]

# ---------------------------------------------------------------- 1. QUERY SERVICE
st.divider()
st.subheader("1. Query Service")
c1, c2, c3 = st.columns(3)
c1.metric("Tipe (berbasis aturan)", TYPE_LABELS.get(qs["ambiguity_type"], "-"))
c2.metric("Istilah pemicu", qs.get("trigger_term") or "-")
c3.metric("Tipe akhir (LLM)", TYPE_LABELS.get(exp.get("detected_type"), "-"))
st.caption(f"Kueri ternormalisasi: *{qs['normalized_query']}*"
           + (f" · {qs['rule']}" if qs.get("rule") else ""))

# ---------------------------------------------------------------- 2. LLM EXPANDER
st.divider()
st.subheader("2. Kandidat interpretasi (LLM Expander)")
st.caption(f"Istilah ambigu: **{exp.get('ambiguous_term')}** · {exp.get('type_reason') or ''}")
scored_by_name = {m["candidate"]["name"]: m for m in state.get("scored", [])}
cand_rows = []
for c in exp["candidates"]:
    m = scored_by_name.get(c["name"])
    um = next((u for u in state["unmapped"] if u["name"] == c["name"]), None)
    cand_rows.append({
        "Kandidat": c["name"], "Jenis konsep": c["entity_type"], "Confidence LLM": c["confidence"],
        "Node PrimeKG": f"{m['node']['name']} [{m['node']['label']}]" if m else "-",
        "Status": STATUS_LABELS["mapped"] if m else STATUS_LABELS.get(um["status"] if um else "", "-"),
        "Cosine (sub-graf)": m["similarity"] if m else None,
        "Alasan": c["reason"]})
st.dataframe(cand_rows, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 3. KLARIFIKASI
if state.get("needs_clarification"):
    st.divider()
    st.subheader("3. Klarifikasi")
    st.info(state.get("clarifying_question") or "Konsep spesifik mana yang Anda maksud?")
    options = {m["node"]["node_index"]: f"{m['node']['name']} [{m['node']['label']}]"
               for m in state["scored"]}
    with st.form("clarify"):
        chosen = st.radio("Pilih konsep yang paling sesuai dengan maksud Anda",
                          list(options), format_func=options.get)
        if st.form_submit_button("Gunakan konsep ini"):
            st.session_state.result = ms.finalize(state, chosen)

res = st.session_state.result
if res is None:
    if not state.get("scored"):
        st.warning("Tidak ada kandidat yang terpetakan ke graf pengetahuan.")
    st.stop()

# ---------------------------------------------------------------- 4. INTERPRETASI
st.divider()
st.subheader("4. Interpretasi terpilih")
ex = res["explanation"]
with st.container(border=True):
    st.markdown(f"### 🎯 {ex['concept']}  <span style='font-size:0.7em;color:gray'>"
                f"{ex['label']}</span>", unsafe_allow_html=True)
    how = "pilihan pengguna (klarifikasi)" if res["selection"] == "clarification" else \
        "cosine similarity tertinggi"
    st.markdown(f"Dipilih berdasarkan **{how}** · skor kemiripan **{ex['similarity']}** · "
                f"relasi utama pada hasil teratas: {ex['main_relations']}")

# ---------------------------------------------------------------- 5. HASIL RETRIEVAL
st.subheader("5. Hasil retrieval (diurutkan dengan cosine similarity)")
st.dataframe([{"Peringkat": n["rank"], "Node": n["name"], "Tipe": n["label"],
               "Relasi": n.get("display_relation") or "(konsep terpilih)",
               "Skor": n["similarity"]} for n in res["ranked_nodes"][:10]],
             use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- 6. VISUALISASI
st.subheader("6. Sub-graf hasil retrieval")
st.caption("Node besar = konsep terpilih; warna menunjukkan tipe node. Arahkan kursor ke "
           "tengah garis untuk melihat relasi.")
st.plotly_chart(subgraph_figure(res["interpretation"], res["ranked_nodes"]),
                use_container_width=True)

with st.expander("Bandingkan dengan baseline (tanpa Semantic Query Expansion)"):
    b = base.run(res["query"])
    st.write("Kata kunci:", ", ".join(b["keywords"]) or "-")
    st.write("Node yang cocok kata kunci:", ", ".join(m["name"] for m in b["matched_nodes"]) or "-")
    if b["interpretation"]:
        st.write(f"Interpretasi baseline: **{b['interpretation']['name']}** "
                 f"[{b['interpretation']['label']}]")
        st.dataframe([{"Peringkat": n["rank"], "Node": n["name"], "Tipe": n["label"],
                       "Skor": n["similarity"]} for n in b["ranked_nodes"][:10]],
                     use_container_width=True, hide_index=True)
    ms.ranker.save_cache()
