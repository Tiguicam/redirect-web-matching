import gradio as gr
import pandas as pd
import re
import time

from rapidfuzz import fuzz
from sentence_transformers import util

# 🔗 On importe le moteur complet
import core_matching as cm


###############################################
# ===========  CONFIG LOCALE UI ==============
###############################################

# On garde une petite CONFIG UI (pour test URL / estimation),
# mais le "vrai" comportement est piloté par core_matching.
CONFIG = {
    "STOPWORDS": set(),
    "EXCEPTIONS": set(),      # on la mappe sur NORMALIZATION_EXCEPTIONS du moteur
    "MULTI_WORD_EXP": [],
    "REFERENCE_REGEX": cm.REFERENCE_PATTERN,
    "TOKEN_BONUS": cm.TOKEN_BONUS_IDENTICAL,
    "TOKEN_MALUS": -cm.TOKEN_PENALTY_EXTRA,
    "SEM_WEIGHT": cm.EMBEDDING_WEIGHT,   # pour l’estimation simple
    "SCORE_MIN": cm.MIN_SCORE_THRESHOLD,
}

# Dico dynamique côté UI (affichage / paramétrage)
DICO_WEIGHTS = {}   # {"dico": {"bonus": int, "malus": int}}
ACTIVE_DICOS = set()  # servira à activer le préfiltre dans le moteur via PREFILTER_ENABLED


###############################################
# ============= UTILS LOCAUX =================
###############################################

def log(msg, logs):
    logs.append(msg)

def normalize_text(txt):
    if not isinstance(txt, str):
        return ""
    return (
        txt.lower()
          .replace("-", " ")
          .replace("_", " ")
          .replace("/", " ")
          .strip()
    )

def clean_url(url):
    """
    Version simplifiée pour le test URL / estimation (pas le moteur).
    """
    u = normalize_text(url)

    for ex in CONFIG["EXCEPTIONS"]:
        u = u.replace(ex, ex)

    for exp in CONFIG["MULTI_WORD_EXP"]:
        u = u.replace(exp.replace("-", " "), exp)

    tokens = u.split()
    tokens = [t for t in tokens if t not in CONFIG["STOPWORDS"]]

    return u, tokens

def apply_translations(tokens):
    # Pour la partie UI simple, on peut utiliser la TRANSLATION_DICT du moteur
    return [cm.TRANSLATION_DICT.get(t, t) for t in tokens]

def apply_dictionaries(tokens):
    """
    Version light : applique ADDITIONAL_DICTS du moteur en remplaçant par la clé canonique.
    Sert uniquement pour le "Tester une URL" et l’estimation, pas pour le vrai matching.
    """
    out = []
    matched = {}
    # Si ACTIVE_DICOS non vide → on limite aux dicos cochés
    if ACTIVE_DICOS:
        active = [name for name in cm.ADDITIONAL_DICTS.keys() if name in ACTIVE_DICOS]
    else:
        active = list(cm.ADDITIONAL_DICTS.keys())

    for t in tokens:
        replaced = False
        for dic_name in active:
            dic_content = cm.ADDITIONAL_DICTS[dic_name]
            for key, syns in dic_content.items():
                if t in syns or t == key:
                    out.append(key)
                    matched.setdefault(dic_name, set()).add(key)
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            out.append(t)
    return out, matched

def extract_reference(tokens):
    rex = CONFIG["REFERENCE_REGEX"]
    for t in tokens:
        m = re.search(rex, t)
        if m:
            return m.group(1)
    return None

def embed_text(text):
    if cm.EMBEDDING_MODEL is None:
        return None
    return cm.EMBEDDING_MODEL.encode(text, convert_to_tensor=True)


###############################################
# ============= ESTIMATION SIMPLE =============
###############################################

def score_pair_estimation(tokens_404, tokens_200):
    """
    Petit scorer simplifié (juste pour estimer le temps).
    Le vrai scoring est dans core_matching.run_matching_core.
    """
    f_score = fuzz.token_set_ratio(" ".join(tokens_404), " ".join(tokens_200))
    emb1 = embed_text(" ".join(tokens_404))
    emb2 = embed_text(" ".join(tokens_200))
    if emb1 is None or emb2 is None:
        sem = 0.0
    else:
        sem = util.cos_sim(emb1, emb2).item()
    return f_score + CONFIG["SEM_WEIGHT"] * sem

def estimate_time_option_B(urls_404, urls_200, logs):

    if not urls_404 or not urls_200:
        log("[ESTIMATION] Impossible : fichiers vides.", logs)
        return None

    # Embedding d'une URL 200
    t0 = time.time()
    embed_text(urls_200[0])
    t200 = time.time() - t0

    # Matching d'une URL 404 (approximation)
    clean404, tok404 = clean_url(urls_404[0])
    tok404 = apply_translations(tok404)
    tok404, _ = apply_dictionaries(tok404)

    clean200, tok200 = clean_url(urls_200[0])
    tok200 = apply_translations(tok200)
    tok200, _ = apply_dictionaries(tok200)

    t1 = time.time()
    score_pair_estimation(tok404, tok200)
    t404 = time.time() - t1

    total = len(urls_404)
    estimated = (t200 + t404) * total

    return estimated


###############################################
# ============ TEST URL (UI) =================
###############################################

def test_url(url: str):
    logs = []

    if not url:
        return "Veuillez saisir une URL."

    # 1) Nettoyage complet EXACT moteur (singularisation, normalisation, https, www, séparateurs…)
    clean = cm.clean_and_normalize_url(url)

    # 2) Substitution synonymes (traductions + dicos dynamiques)
    sub = cm.apply_synonym_substitution(clean)

    # 3) Tokens (après stopwords, exceptions, multi-mots…)
    tokens = cm.get_tokens(sub)

    # 4) Référence produit via la regex exacte du moteur
    ref = cm.extract_reference(url)

    # 5) Détection des caractéristiques via les dicos dynamiques EXACT moteur
    dico_hits = {}
    for dico_name, dico_dict in cm.ADDITIONAL_DICTS.items():
        hits = cm.detect_characteristic(sub, dico_dict)
        dico_hits[dico_name] = hits

    # ===========================
    # LOGS
    # ===========================

    log(f"URL brute : {url}", logs)
    log(f"URL nettoyée (clean_and_normalize_url) :", logs)
    log(f" → {clean}", logs)

    log("URL après synonymes (apply_synonym_substitution) :", logs)
    log(f" → {sub}", logs)

    log("Tokens (get_tokens) :", logs)
    if tokens:
        log(" → " + ", ".join(tokens), logs)
    else:
        log(" → Aucun token", logs)

    if ref:
        log(f"Référence détectée : {ref}", logs)
    else:
        log("Aucune référence détectée via la regex.", logs)

    log("Caractéristiques détectées :", logs)
    if not dico_hits:
        log(" → Aucun dictionnaire dynamique chargé.", logs)
    else:
        for dico_name, keys in dico_hits.items():
            if keys:
                log(f" [{dico_name}] → {', '.join(keys)}", logs)
            else:
                log(f" [{dico_name}] → aucune caractéristique trouvée", logs)

    return "\n".join(logs)


###############################################
# =========== CONFIG UPDATE ==================
###############################################

def build_dico_weights_from_file(path):
    """
    Construit un DataFrame avec une ligne par dico (colonne de l'Excel)
    et des colonnes Dictionnaire / Bonus / Malus / Actif.
    """
    if path is None:
        return pd.DataFrame(columns=["Dictionnaire", "Bonus", "Malus", "Actif"])

    df = pd.read_excel(path.name)
    rows = []
    for col in df.columns:
        rows.append([
            col,
            CONFIG["TOKEN_BONUS"],         # bonus par défaut
            abs(CONFIG["TOKEN_MALUS"]),   # malus affiché en positif
            True                          # Actif (coché) = préfiltre ON
        ])
    return pd.DataFrame(rows, columns=["Dictionnaire", "Bonus", "Malus", "Actif"])


def update_config(dico, trad, parents, stop, exc, multi, ref, tbonus, tmalus, semw, minscore, dico_dyn):
    """
    Met à jour :
      - la petite CONFIG locale (UI)
      - et TOUTE la config du moteur core_matching :
          STOPWORDS, MULTI_WORD_EXPRESSIONS, NORMALIZATION_EXCEPTIONS,
          REFERENCE_PATTERN, TOKEN_BONUS_IDENTICAL, TOKEN_PENALTY_EXTRA,
          EMBEDDING_WEIGHT, MIN_SCORE_THRESHOLD,
          RAW_TRANSLATION_DICT -> renormalize_translation_dict(),
          RAW_ADDITIONAL_DICTS -> renormalize_additional_dicts(),
          set_parent_urls(),
          ADDITIONAL_BONUS_VALUES, ADDITIONAL_MALUS_VALUES, PREFILTER_ENABLED.
    """
    logs = []

    # ==== 1) CONFIG UI ====
    CONFIG["STOPWORDS"] = {s.strip().lower() for s in stop.split(",") if s.strip()}
    CONFIG["EXCEPTIONS"] = {s.strip().lower() for s in exc.split(",") if s.strip()}
    CONFIG["MULTI_WORD_EXP"] = [s.strip().lower() for s in multi.split(",") if s.strip()]
    CONFIG["REFERENCE_REGEX"] = ref
    CONFIG["TOKEN_BONUS"] = int(tbonus)
    CONFIG["TOKEN_MALUS"] = -abs(int(tmalus))
    CONFIG["SEM_WEIGHT"] = float(semw)
    CONFIG["SCORE_MIN"] = float(minscore)

    # ==== 2) Pousser vers le moteur core_matching ====
    # Stopwords / multi-mots / exceptions / regex
    cm.STOPWORDS.clear()
    cm.STOPWORDS.update(CONFIG["STOPWORDS"])

    cm.MULTI_WORD_EXPRESSIONS.clear()
    cm.MULTI_WORD_EXPRESSIONS.extend(CONFIG["MULTI_WORD_EXP"])

    cm.NORMALIZATION_EXCEPTIONS.clear()
    cm.NORMALIZATION_EXCEPTIONS.update(CONFIG["EXCEPTIONS"])

    cm.REFERENCE_PATTERN = CONFIG["REFERENCE_REGEX"]

    # Pondérations globales
    cm.TOKEN_BONUS_IDENTICAL = CONFIG["TOKEN_BONUS"]
    cm.TOKEN_PENALTY_EXTRA = CONFIG["TOKEN_MALUS"]
    cm.EMBEDDING_WEIGHT = float(semw)
    cm.MIN_SCORE_THRESHOLD = float(minscore)

    # ==== 3) Traductions → RAW_TRANSLATION_DICT ====
    cm.RAW_TRANSLATION_DICT.clear()
    if trad is not None:
        df_tr = pd.read_excel(trad.name)
        if not {"SOURCE", "TRANSLATION"}.issubset(df_tr.columns):
            log("[TRAD] Colonnes SOURCE / TRANSLATION manquantes dans le fichier traductions.", logs)
        else:
            for _, row in df_tr.iterrows():
                src = str(row["SOURCE"]).strip().lower()
                tr = str(row["TRANSLATION"]).strip().lower()
                if src:
                    cm.RAW_TRANSLATION_DICT[src] = tr
            log(f"[TRAD] {len(cm.RAW_TRANSLATION_DICT)} entrées de traduction brutes chargées.", logs)

    # Appliquer la table de traduction brute → TRANSLATION_DICT
    cm.renormalize_translation_dict()
    log("[TRAD] Table de traduction normalisée (TRANSLATION_DICT) reconstruite.", logs)

    # ==== 4) Dictionnaire Excel → RAW_ADDITIONAL_DICTS ====
    cm.RAW_ADDITIONAL_DICTS.clear()
    if dico is not None:
        df_dico = pd.read_excel(dico.name)
        for dico_name in df_dico.columns:
            dico_content = {}
            for cell_value in df_dico[dico_name].dropna():
                items = [str(it).strip().lower() for it in str(cell_value).split(",") if str(it).strip()]
                if not items:
                    continue
                key = items[0]
                synonyms = set(items)
                if key in dico_content:
                    dico_content[key] |= synonyms
                else:
                    dico_content[key] = synonyms
            cm.RAW_ADDITIONAL_DICTS[dico_name] = dico_content
        log(f"[DICO] Dictionnaires bruts importés depuis l'Excel ({len(df_dico.columns)} colonnes).", logs)

    # Appliquer traduction + normalisation → ADDITIONAL_DICTS
    cm.renormalize_additional_dicts()
    log("[DICO] Dictionnaires dynamiques normalisés (ADDITIONAL_DICTS) reconstruits.", logs)

    # ==== 5) Parents → set_parent_urls ====
    if parents is None:
        cm.set_parent_urls([], None)
        log("[PARENTS] Aucun fichier parents fourni.", logs)
    else:
        df_par = pd.read_excel(parents.name)
        if "URL" not in df_par.columns:
            log("[PARENTS] Colonne URL manquante dans le fichier parents.", logs)
        else:
            url_list = df_par["URL"].dropna().astype(str).tolist()
            cm.set_parent_urls(url_list, source_file=parents.name)
            log(f"[PARENTS] {len(url_list)} URLs parents chargées pour le moteur.", logs)

    # ==== 6) Bonus/Malus par dico + Préfiltre (Actif) ====
    global DICO_WEIGHTS, ACTIVE_DICOS
    DICO_WEIGHTS = {}
    ACTIVE_DICOS = set()

    cm.ADDITIONAL_BONUS_VALUES.clear()
    cm.ADDITIONAL_MALUS_VALUES.clear()
    cm.PREFILTER_ENABLED.clear()

    if dico_dyn is not None:
        if not isinstance(dico_dyn, pd.DataFrame):
            try:
                dico_dyn = pd.DataFrame(dico_dyn, columns=["Dictionnaire", "Bonus", "Malus", "Actif"])
            except Exception:
                dico_dyn = pd.DataFrame(columns=["Dictionnaire", "Bonus", "Malus", "Actif"])

        if not dico_dyn.empty:
            for _, row in dico_dyn.iterrows():
                name = str(row.get("Dictionnaire", "")).strip()
                if not name:
                    continue

                try:
                    bonus = int(row.get("Bonus", CONFIG["TOKEN_BONUS"]))
                except Exception:
                    bonus = CONFIG["TOKEN_BONUS"]

                try:
                    malus_raw = int(row.get("Malus", abs(CONFIG["TOKEN_MALUS"])))
                except Exception:
                    malus_raw = abs(CONFIG["TOKEN_MALUS"])
                malus = -abs(malus_raw)

                actif_val = row.get("Actif", True)
                if isinstance(actif_val, str):
                    actif = actif_val.strip().lower() in ("true", "1", "oui", "yes", "y", "o")
                else:
                    actif = bool(actif_val)

                # Stockage pour l’UI
                DICO_WEIGHTS[name] = {"bonus": bonus, "malus": malus}
                if actif:
                    ACTIVE_DICOS.add(name)

                # Et on pousse dans le moteur :
                cm.ADDITIONAL_BONUS_VALUES[name] = bonus
                cm.ADDITIONAL_MALUS_VALUES[name] = abs(malus)
                cm.PREFILTER_ENABLED[name] = actif

    log("[CONFIG] Paramètres UI + moteur mis à jour.", logs)
    if cm.ADDITIONAL_BONUS_VALUES:
        log(f"[CONFIG] Bonus par dico (moteur) : {cm.ADDITIONAL_BONUS_VALUES}", logs)
    if cm.ADDITIONAL_MALUS_VALUES:
        log(f"[CONFIG] Malus par dico (moteur) : {cm.ADDITIONAL_MALUS_VALUES}", logs)
    if cm.PREFILTER_ENABLED:
        log(f"[CONFIG] Préfiltre actif (PREFILTER_ENABLED) : {cm.PREFILTER_ENABLED}", logs)

    return "\n".join(logs)


###############################################
# ============== UI : MATCHING ===============
###############################################

def afficher_estimation(f404, f200):
    logs = []

    if f404 is None or f200 is None:
        return ""

    df404 = pd.read_excel(f404.name, sheet_name="404")
    df200 = pd.read_excel(f200.name, sheet_name="200")

    urls_404 = df404["URL_404"].astype(str).tolist()
    urls_200 = df200["URL_200"].astype(str).tolist()

    est = estimate_time_option_B(urls_404, urls_200, logs)
    if est:
        log(f"[ESTIMATION] ~ {est/60:.1f} minutes (approximation, moteur complet un peu plus long)", logs)
    else:
        log("[ESTIMATION] Impossible d'estimer.", logs)

    return "\n".join(logs)


def lancer_matching(f404, f200, progress=gr.Progress()):
    logs = []

    if f404 is None or f200 is None:
        return "Upload les fichiers d'abord.", None

    # Progress callback pour récupérer les infos du moteur
    def progress_cb(pct, msg):
        # pct = 0–100 → on normalise entre 0 et 1 pour la barre
        try:
            progress(pct / 100, desc=msg)
        except Exception:
            # au cas où ta version de Gradio n'accepte pas desc=
            progress(pct / 100)
        logs.append(f"[{pct}%] {msg}")

    try:
        df_result, debug_df, output_path, core_logs, times = cm.run_matching_core(
            f404.name,
            f200.name,
            output_file="result.xlsx",
            debug_mode=True,
            create_hist=False,   # éviter plt.show() côté serveur
            progress_cb=progress_cb,
        )
    except Exception as e:
        logs.append(f"[ERREUR] {e}")
        return "\n".join(logs), None

    # Fusion logs internes moteur + logs progression
    all_logs = logs + core_logs
    return "\n".join(all_logs), "result.xlsx"


###############################################
# ================== UI GRADIO ===============
###############################################

with gr.Blocks(title="Matching 404/200 PRO (Gradio + core_matching)") as demo:

    with gr.Tab("Configuration"):
        dico = gr.File(label="Dictionnaire Excel")
        trad = gr.File(label="Fichier Traductions")
        parents = gr.File(label="Parents (optionnel)")

        stop = gr.Textbox(label="Stopwords (virgules)")
        exc = gr.Textbox(label="Exceptions normalisation (virgules)")
        multi = gr.Textbox(label="Expressions multi-mots (virgules)")

        ref = gr.Textbox(label="Regex référence", value=CONFIG["REFERENCE_REGEX"])
        tbonus = gr.Number(label="Token + (bonus identiques)", value=CONFIG["TOKEN_BONUS"])
        tmalus = gr.Number(label="Token - (malus extra tokens)", value=abs(CONFIG["TOKEN_MALUS"]))
        semw = gr.Number(label="Poids sémantique (moteur)", value=cm.EMBEDDING_WEIGHT)
        minscore = gr.Number(label="Score minimum (moteur)", value=cm.MIN_SCORE_THRESHOLD)

        # Tableau dynamique : bonus/malus + préfiltre (Actif)
        dico_dyn = gr.Dataframe(
            headers=["Dictionnaire", "Bonus", "Malus", "Actif"],
            label="Dico dynamique (bonus/malus + préfiltre par dico)",
            row_count=(0, "dynamic"),
            datatype=["str", "number", "number", "bool"],
        )

        # Remplir à partir de l'Excel de dico
        dico.change(
            fn=build_dico_weights_from_file,
            inputs=dico,
            outputs=dico_dyn,
        )

        btn = gr.Button("Mettre à jour la configuration (moteur)")
        out = gr.Textbox(label="Logs configuration", lines=12)

        btn.click(
            update_config,
            inputs=[dico, trad, parents, stop, exc, multi, ref, tbonus, tmalus, semw, minscore, dico_dyn],
            outputs=out,
        )

    with gr.Tab("Tester une URL"):
        url = gr.Textbox(label="URL")
        btn_test = gr.Button("Tester")
        res_test = gr.Textbox(label="Résultat", lines=12)

        btn_test.click(test_url, inputs=url, outputs=res_test)

    with gr.Tab("Matching"):
        f404 = gr.File(label="Fichier 404 (sheet '404', col 'URL_404')")
        f200 = gr.File(label="Fichier 200 (sheet '200', col 'URL_200')")

        logs_est = gr.Textbox(label="Estimation (approx.)", lines=4)

        f404.change(afficher_estimation, inputs=[f404, f200], outputs=logs_est)
        f200.change(afficher_estimation, inputs=[f404, f200], outputs=logs_est)

        btn_run = gr.Button("Lancer le matching (moteur core_matching)")
        logs_match = gr.Textbox(label="Logs matching", lines=25)
        dl = gr.File(label="Télécharger résultat")

        btn_run.click(
            lancer_matching,
            inputs=[f404, f200],
            outputs=[logs_match, dl],
        )

demo.launch()
