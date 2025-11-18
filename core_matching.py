import os
import re
import time
import pickle
import hashlib
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt

# Embeddings
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None
    EMBEDDING_MODEL = None
else:
    try:
        EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception as e:
        print(f"Impossible de charger le modèle d'embedding : {e}")
        EMBEDDING_MODEL = None


# ===================================================================
# CONFIG GLOBALE / ÉTAT
# ===================================================================

DEBUG_MODE = True

# Parents (pré-calculs rapides)
PARENT_URL_LIST: list[str] = []
PARENT_URL_EMBEDDINGS: list[np.ndarray] = []
PARENT_MATRIX: np.ndarray | None = None
PARENT_CLEAN: list[str] = []
PARENT_TOKENS: list[set[str]] = []
PARENT_SOURCE_FILE: str | None = None
PARENTS_PREPARED: bool = False

# Dictionnaires dynamiques
ADDITIONAL_DICTS: dict[str, dict[str, set[str]]] = {}
RAW_ADDITIONAL_DICTS: dict[str, dict[str, set[str]]] = {}

# Table de traduction
TRANSLATION_DICT: dict[str, str] = {}
RAW_TRANSLATION_DICT: dict[str, str] = {}

# Exceptions de normalisation
NORMALIZATION_EXCEPTIONS: set[str] = set()

# Multi-mots & stopwords
MULTI_WORD_EXPRESSIONS: list[str] = []
STOPWORDS: set[str] = set()

# Pondérations globales
TOKEN_BONUS_IDENTICAL: int = 5
TOKEN_PENALTY_EXTRA: int = -10
EMBEDDING_WEIGHT: float = 80.0
MIN_SCORE_THRESHOLD: float = 20.0

# Regex de référence produit
REFERENCE_PATTERN: str = r"[A-Za-z]?\d{6,7}-\d{1,2}"

# Caches embeddings pour 404 / 200
CACHE_404: dict[str, np.ndarray] = {}
CACHE_200: dict[str, np.ndarray] = {}

# Pondérations par dico (côté moteur, simple ints/bools)
ADDITIONAL_BONUS_VALUES: dict[str, int] = {}     # ex: {"genre": 10, "couleur": 5}
ADDITIONAL_MALUS_VALUES: dict[str, int] = {}     # ex: {"genre": 20, "couleur": 10}
PREFILTER_ENABLED: dict[str, bool] = {}          # ex: {"genre": True, "couleur": False}


# ===================================================================
# UTILITAIRES TEMPS
# ===================================================================

def tick(times_dict: dict, key: str) -> None:
    times_dict[key] = time.perf_counter()


def elapsed(times_dict: dict, start_key: str, end_key: str) -> float:
    return times_dict.get(end_key, 0) - times_dict.get(start_key, 0)


# ===================================================================
# NORMALISATION / NETTOYAGE
# ===================================================================

def normalize_word(word: str) -> str:
    """
    Exceptions → tag langue → traduction brute → singularisation → traduction singulier → fallback.
    """
    global TRANSLATION_DICT, NORMALIZATION_EXCEPTIONS
    w0 = (word or "").lower().strip()

    if w0 in NORMALIZATION_EXCEPTIONS:
        return w0

    # Tag langue (fr, fr-fr, en-us, etc.)
    if re.fullmatch(r'[a-z]{2,3}(?:-[a-z0-9]{2,4}){0,2}', w0):
        return w0

    # 1) Traduction brute
    if w0 in TRANSLATION_DICT:
        return TRANSLATION_DICT[w0]

    # 2) Singularisation très simple
    w1 = w0[:-1] if w0.endswith("s") else w0

    # 3) Traduction sur forme singularisée
    if w1 in TRANSLATION_DICT:
        return TRANSLATION_DICT[w1]

    return w1


def clean_and_normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    url = url.lower()

    # Préserver les expressions multi-mots
    for expression in MULTI_WORD_EXPRESSIONS:
        if expression in url:
            url = url.replace(expression, expression.replace("-", "*"))

    # Séparateurs → espaces
    for sep in [".html", ".htm", "/", "-", "_", ".", ",", ":", "!", "?"]:
        url = url.replace(sep, " ")
    url = " ".join(url.split())

    normalized_tokens = []
    for token in url.split():
        token = token.replace("*", "-")
        normalized_tokens.append(normalize_word(token))
    return " ".join(normalized_tokens)


def extract_reference(url: str) -> str:
    if not url:
        return ""
    pattern = REFERENCE_PATTERN
    match = re.search(pattern, url, re.IGNORECASE)
    if match:
        return match.group(0).strip().lower()
    return ""


def get_parent_url(url: str) -> str:
    """
    Retourne l’URL parent (un niveau au-dessus) en gardant le slash final.
    Ex.  https://site.com/collection/pantalon/rouge
    →    https://site.com/collection/pantalon/
    """
    if not isinstance(url, str):
        return ""
    sc, net, path, *_ = urlsplit(url)
    path = path.rstrip("/")
    parent_path = "/".join(path.split("/")[:-1]) + "/"
    return urlunsplit((sc, net, parent_path, "", ""))


# ===================================================================
# CHARGEMENT / FEATURES
# ===================================================================

def load_data(file_404, file_200):
    """
    Charge Excel sheet_name=404 & 200, applique clean & ref.
    Ajoute une colonne REF_DETECTED = OUI/NON.
    """
    df404 = pd.read_excel(file_404, sheet_name="404")
    df200 = pd.read_excel(file_200, sheet_name="200")

    df404["clean_404"] = df404["URL_404"].apply(clean_and_normalize_url)
    df200["clean_200"] = df200["URL_200"].apply(clean_and_normalize_url)

    df404["ref_404"] = df404["URL_404"].apply(extract_reference)
    df200["ref_200"] = df200["URL_200"].apply(extract_reference)

    df404["REF_DETECTED"] = df404["ref_404"].apply(lambda x: "OUI" if x else "NON")
    df200["REF_DETECTED"] = df200["ref_200"].apply(lambda x: "OUI" if x else "NON")
    return df404, df200


def detect_characteristic(text: str, dictionary: dict) -> str:
    """
    Détecte toutes les 'keys' dont au moins un synonyme est présent dans 'text'.
    Retourne une chaîne "key1,key2" si plusieurs clés matchent, ou "" si aucune.
    """
    words = set((text or "").split())
    matched_keys = []
    for key, synonyms in dictionary.items():
        if words & synonyms:
            matched_keys.append(key)
    return ",".join(matched_keys)


def add_dynamic_characteristics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Pour chaque dictionnaire (colonne) dans ADDITIONAL_DICTS,
    crée une colonne ex: 'categorie_404' où sont stockées toutes les clés matchées.
    """
    for dico_name, dico_data in ADDITIONAL_DICTS.items():
        col_name = f"{dico_name.lower()}_{prefix}"
        df[col_name] = df[f"clean_{prefix}"].apply(lambda x: detect_characteristic(x, dico_data))
    return df


# ===================================================================
# TOKENS / SCORE
# ===================================================================

def get_tokens(text: str) -> set:
    if not text:
        return set()
    return {t for t in text.split() if t not in STOPWORDS}


def exact_token_bonus(tokens_404: set, tokens_200: set) -> int:
    return TOKEN_BONUS_IDENTICAL * len(tokens_404 & tokens_200)


def extra_tokens_malus(tokens_404: set, tokens_200: set) -> int:
    return TOKEN_PENALTY_EXTRA * len(tokens_200 - tokens_404)


# ===================================================================
# EMBEDDINGS
# ===================================================================

def compute_embedding(text: str):
    if EMBEDDING_MODEL is None:
        return None
    return EMBEDDING_MODEL.encode(text)


def compute_embeddings_batch(texts: list[str]):
    if EMBEDDING_MODEL is None:
        return [None] * len(texts)
    return EMBEDDING_MODEL.encode(texts, batch_size=64, show_progress_bar=False)


def cosine_similarity(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    score = util.cos_sim(vec1, vec2)
    return float(score[0][0])


def batch_embeddings(df: pd.DataFrame, col_name: str, cache_dict: dict) -> pd.Series:
    """
    Encode une colonne texte en embeddings, avec cache simple.
    """
    if EMBEDDING_MODEL is None:
        return pd.Series([None] * len(df))

    texts = df[col_name].tolist()
    to_encode = []
    positions = []
    results = [None] * len(texts)

    for i, t in enumerate(texts):
        if t in cache_dict:
            results[i] = cache_dict[t]
        else:
            to_encode.append(t)
            positions.append(i)

    batch_size = 64
    for start in range(0, len(to_encode), batch_size):
        batch = to_encode[start:start + batch_size]
        embs = EMBEDDING_MODEL.encode(batch, batch_size=batch_size, show_progress_bar=False)
        for j, emb in enumerate(embs):
            idx = positions[start + j]
            cache_dict[texts[idx]] = emb
            results[idx] = emb

    return pd.Series(results)


# ===================================================================
# DICTIONNAIRES / TRADUCTION
# ===================================================================

def renormalize_additional_dicts():
    """
    Reconstruit ADDITIONAL_DICTS à partir de RAW_ADDITIONAL_DICTS
    en appliquant : traduction (via normalize_word) → normalisation → fusion.
    """
    global ADDITIONAL_DICTS, RAW_ADDITIONAL_DICTS
    new_dicts = {}

    for dico_name, raw_dico in RAW_ADDITIONAL_DICTS.items():
        rebuilt = {}
        for raw_key, raw_syns in raw_dico.items():
            all_items = [raw_key] + list(raw_syns)
            norm_items = [normalize_word(x) for x in all_items if x]

            if not norm_items:
                continue

            nk = norm_items[0]
            ns = set(norm_items)

            if nk in rebuilt:
                rebuilt[nk] |= ns
            else:
                rebuilt[nk] = ns

        new_dicts[dico_name] = rebuilt

    ADDITIONAL_DICTS = new_dicts


def renormalize_translation_dict():
    """
    Reconstruit TRANSLATION_DICT à partir de RAW_TRANSLATION_DICT,
    en respectant les exceptions. Pas d'appel à normalize_word ici.
    """
    global TRANSLATION_DICT, RAW_TRANSLATION_DICT, NORMALIZATION_EXCEPTIONS
    new_trans = {}

    for raw_src, raw_dst in RAW_TRANSLATION_DICT.items():
        src = (raw_src or "").strip().lower()
        dst = (raw_dst or "").strip().lower()
        if not src:
            continue
        if src in NORMALIZATION_EXCEPTIONS:
            continue
        new_trans[src] = dst

    TRANSLATION_DICT = new_trans


def apply_synonym_substitution(text: str) -> str:
    """
    Pour chaque token dans le texte, si ce token figure dans l'un des dictionnaires additionnels,
    le remplace par sa clé canonique.
    """
    tokens = (text or "").split()
    new_tokens = []
    for token in tokens:
        t = token.lower()
        replaced = False
        for dico in ADDITIONAL_DICTS.values():
            for canonical, synonyms in dico.items():
                if t in synonyms or t == canonical:
                    new_tokens.append(canonical)
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            new_tokens.append(t)
    return " ".join(new_tokens)


# ===================================================================
# PARENTS
# ===================================================================

def set_parent_urls(url_list: list[str], source_file: str | None = None):
    """
    À appeler côté UI pour définir la liste brute des URLs parents.
    """
    global PARENT_URL_LIST, PARENT_SOURCE_FILE, PARENTS_PREPARED
    PARENT_URL_LIST = [u for u in url_list if isinstance(u, str)]
    PARENT_SOURCE_FILE = source_file
    PARENTS_PREPARED = False


def prepare_parents_artifacts():
    """
    Nettoie + substitue + tokenise + charge/calcule embeddings parents,
    et construit PARENT_MATRIX.
    """
    global PARENT_CLEAN, PARENT_TOKENS, PARENT_URL_EMBEDDINGS, PARENT_MATRIX, PARENTS_PREPARED

    if not PARENT_URL_LIST:
        PARENT_CLEAN, PARENT_TOKENS, PARENT_URL_EMBEDDINGS, PARENT_MATRIX = [], [], [], None
        PARENTS_PREPARED = True
        return

    # 1) Clean + substitution
    PARENT_CLEAN = [apply_synonym_substitution(clean_and_normalize_url(u)) for u in PARENT_URL_LIST]
    PARENT_TOKENS = [get_tokens(c) for c in PARENT_CLEAN]

    # 2) Cache embeddings parents
    parents_fingerprint = hashlib.sha1("\n".join(PARENT_CLEAN).encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(PARENT_SOURCE_FILE))[0] if PARENT_SOURCE_FILE else "parents"
    cache_file = f"{base}.{parents_fingerprint}.pkl"

    loaded = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, list) and len(cached) == len(PARENT_CLEAN) and all(c is not None for c in cached):
                PARENT_URL_EMBEDDINGS = cached
                loaded = True
        except Exception:
            loaded = False

    if not loaded:
        if EMBEDDING_MODEL:
            PARENT_URL_EMBEDDINGS = [None] * len(PARENT_CLEAN)
            bs = 64
            for i in range(0, len(PARENT_CLEAN), bs):
                batch = PARENT_CLEAN[i:i + bs]
                embs = EMBEDDING_MODEL.encode(batch, batch_size=bs, show_progress_bar=False)
                PARENT_URL_EMBEDDINGS[i:i + bs] = list(embs)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(PARENT_URL_EMBEDDINGS, f)
            except Exception:
                pass
        else:
            PARENT_URL_EMBEDDINGS = [None] * len(PARENT_CLEAN)

    if EMBEDDING_MODEL and len(PARENT_URL_EMBEDDINGS) > 0 and all(e is not None for e in PARENT_URL_EMBEDDINGS):
        PARENT_MATRIX = np.vstack(PARENT_URL_EMBEDDINGS).astype(np.float32)
        PARENT_MATRIX /= (np.linalg.norm(PARENT_MATRIX, axis=1, keepdims=True) + 1e-9)
    else:
        PARENT_MATRIX = None

    PARENTS_PREPARED = True


# ===================================================================
# MATCHING PRINCIPAL (MOTEUR PUR)
# ===================================================================

def run_matching_core(
    file_404,
    file_200,
    *,
    output_file: str = "match_404_200_result.xlsx",
    debug_mode: bool = True,
    create_hist: bool = True,
    progress_cb=None,
):
    """
    Version 'moteur' de run_matching :
      - pas de Tkinter
      - pas de messagebox
      - ne dépend pas de widgets
    Retourne :
      df_result, debug_df (ou None), output_file, logs (liste de str), times (dict)
    """
    global DEBUG_MODE
    DEBUG_MODE = debug_mode

    logs: list[str] = []
    debug_details: list[dict] = []
    times: dict[str, float] = {}

    def _log(msg: str):
        logs.append(msg)

    def _progress(pct: int, msg: str):
        if progress_cb is not None:
            progress_cb(int(pct), msg)

    tick(times, "start_all")

    # Préparation parents si besoin
    if PARENT_URL_LIST and not PARENTS_PREPARED:
        prepare_parents_artifacts()

    # Chargement
    try:
        tick(times, "t_load_start")
        df404, df200 = load_data(file_404, file_200)
        df404 = df404.reset_index(drop=True)
        df200 = df200.reset_index(drop=True)
        tick(times, "t_load_end")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des fichiers: {e}") from e

    _progress(5, "Chargement des données terminé")

    # Caractéristiques dynamiques
    tick(times, "t_dyn_start")
    df404 = add_dynamic_characteristics(df404, "404")
    df200 = add_dynamic_characteristics(df200, "200")
    tick(times, "t_dyn_end")
    _progress(10, "Extraction des caractéristiques")

    # Substitution + tokens
    tick(times, "t_subs_start")
    df404["sub_clean_404"] = df404["clean_404"].apply(apply_synonym_substitution)
    df200["sub_clean_200"] = df200["clean_200"].apply(apply_synonym_substitution)
    df200["tokens_200"] = df200["sub_clean_200"].apply(get_tokens)
    tick(times, "t_subs_end")
    _progress(15, "Substitution des synonymes")

    # Embeddings
    if EMBEDDING_MODEL is not None:
        tick(times, "t_emb_start")
        df404["embedding_404"] = batch_embeddings(df404, "sub_clean_404", CACHE_404)
        df200["embedding_200"] = batch_embeddings(df200, "sub_clean_200", CACHE_200)

        matrix_200 = np.vstack(df200["embedding_200"].tolist()).astype(np.float32)
        norms = np.linalg.norm(matrix_200, axis=1, keepdims=True) + 1e-9
        matrix_200 = matrix_200 / norms

        def topk_similarity_for_row(emb_404_vec: np.ndarray, embs_200_norm: np.ndarray, k: int = 500):
            if emb_404_vec is None:
                return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
            v = emb_404_vec.astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-9)
            sims = embs_200_norm @ v
            m = sims.shape[0]
            k = min(k, m)
            idx = np.argpartition(sims, -k)[-k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
            return idx.astype(np.int32), sims[idx].astype(np.float32)

        tick(times, "t_emb_end")
        _progress(30, "Encodage embeddings terminé")
    else:
        df404["embedding_404"] = None
        df200["embedding_200"] = None
        matrix_200 = None
        topk_similarity_for_row = None  # type: ignore

    # Boucle principale
    best_matches = []
    total_404 = len(df404)
    if total_404 == 0:
        raise ValueError("Aucune URL 404 trouvée dans le fichier 404.")

    tick(times, "t_loop_start")

    for i, (_, row_404) in enumerate(df404.iterrows()):
        url_404_sub = row_404["sub_clean_404"]
        embedding_404 = row_404["embedding_404"]
        tokens_404 = get_tokens(url_404_sub)
        ref_404 = row_404["ref_404"]

        same_ref_scored = []
        used_same_ref = False

        if EMBEDDING_MODEL is not None and embedding_404 is not None and matrix_200 is not None:
            v404 = embedding_404.astype(np.float32)
            v404 /= (np.linalg.norm(v404) + 1e-9)
        else:
            v404 = None

        # Cas 1 : REF identique
        if ref_404:
            same_ref_df = df200[df200["ref_200"] == ref_404]
            if not same_ref_df.empty:
                used_same_ref = True
                for _, row_200 in same_ref_df.iterrows():
                    tokens_200 = row_200["tokens_200"]

                    score = 0

                    # Dicos dynamiques
                    for dico_name_ in ADDITIONAL_DICTS.keys():
                        col_404_name = f"{dico_name_.lower()}_404"
                        col_200_name = f"{dico_name_.lower()}_200"
                        val_404_d = row_404.get(col_404_name, "")
                        val_200_d = row_200.get(col_200_name, "")

                        bonus_val = int(ADDITIONAL_BONUS_VALUES.get(dico_name_, 0))
                        malus_val = int(ADDITIONAL_MALUS_VALUES.get(dico_name_, 0))

                        set_404 = set(val_404_d.split(",")) if val_404_d else set()
                        set_200 = set(val_200_d.split(",")) if val_200_d else set()

                        if set_404 and set_200:
                            if set_404 & set_200:
                                score += bonus_val
                            else:
                                score -= abs(malus_val)
                        else:
                            score -= abs(malus_val)

                    # Tokens
                    score += exact_token_bonus(tokens_404, tokens_200)
                    score += extra_tokens_malus(tokens_404, tokens_200)

                    # Réf identique
                    score += 1000

                    # Embedding
                    if v404 is not None:
                        sim_embedding = float(matrix_200[row_200.name] @ v404)
                    else:
                        sim_embedding = 0.0

                    total_score = score + (sim_embedding * EMBEDDING_WEIGHT)

                    if DEBUG_MODE:
                        debug_details.append({
                            "URL_404": row_404["URL_404"],
                            "URL_200_candidate": row_200["URL_200"],
                            "Fuzzy_score": 0,
                            "Score_avant_embedding": score,
                            "Similarité_embedding": sim_embedding,
                            "Score_final": total_score,
                        })

                    same_ref_scored.append((row_200["URL_200"], total_score))

                same_ref_scored = [m for m in same_ref_scored if m[1] >= MIN_SCORE_THRESHOLD]
                same_ref_scored.sort(key=lambda x: x[1], reverse=True)

        # Cas 2 : pas / pas seulement REF
        if used_same_ref and len(same_ref_scored) > 0:
            top_match_1 = same_ref_scored[0] if len(same_ref_scored) > 0 else (None, 0)
            top_match_2 = same_ref_scored[1] if len(same_ref_scored) > 1 else (None, 0)
        else:
            if EMBEDDING_MODEL is not None and embedding_404 is not None and matrix_200 is not None:
                K = max(500, int(0.05 * len(df200)))
                top_idx, top_sims = topk_similarity_for_row(embedding_404, matrix_200, k=K)  # type: ignore
                semantic_set = set(map(int, top_idx))
                idx_to_sim = {int(i): float(s) for i, s in zip(top_idx, top_sims)}
            else:
                semantic_set = set(df200.index)
                idx_to_sim = {}

            # REF identique
            same_ref_idx = set()
            if ref_404:
                same_ref_idx = set(df200.index[df200["ref_200"] == ref_404])

            # Préfiltres dicos cochés
            prefilter_idx = set()
            for dico_name, _ in ADDITIONAL_DICTS.items():
                if PREFILTER_ENABLED.get(dico_name, False):
                    val_404 = row_404.get(f"{dico_name.lower()}_404", "")
                    if val_404:
                        subset_idx = df200.index[df200[f"{dico_name.lower()}_200"] == val_404]
                        prefilter_idx |= set(subset_idx)

            candidate_idx = (semantic_set | same_ref_idx | prefilter_idx)
            df200_filtered = df200.loc[sorted(candidate_idx)].copy()

            candidates_200 = df200_filtered["sub_clean_200"].tolist()
            raw_matches = process.extract(
                url_404_sub,
                candidates_200,
                scorer=fuzz.token_set_ratio,
                limit=100,
            )

            detailed_scored = []
            for match_text, fuzzy_score, idx_in_list in raw_matches:
                row_200 = df200_filtered.iloc[idx_in_list]
                tokens_200 = row_200["tokens_200"]

                score = fuzzy_score

                for dico_name_ in ADDITIONAL_DICTS.keys():
                    col_404_name = f"{dico_name_.lower()}_404"
                    col_200_name = f"{dico_name_.lower()}_200"
                    val_404_d = row_404.get(col_404_name, "")
                    val_200_d = row_200.get(col_200_name, "")

                    bonus_val = int(ADDITIONAL_BONUS_VALUES.get(dico_name_, 0))
                    malus_val = int(ADDITIONAL_MALUS_VALUES.get(dico_name_, 0))

                    set_404 = set(val_404_d.split(",")) if val_404_d else set()
                    set_200 = set(val_200_d.split(",")) if val_200_d else set()

                    if set_404 and set_200:
                        if set_404 & set_200:
                            score += bonus_val
                        else:
                            score -= abs(malus_val)
                    else:
                        score -= abs(malus_val)

                score += exact_token_bonus(tokens_404, tokens_200)
                score += extra_tokens_malus(tokens_404, tokens_200)

                ref_200 = row_200["ref_200"]
                if ref_404 and ref_200 and ref_404 == ref_200:
                    score += 1000

                sim_embedding = idx_to_sim.get(int(row_200.name), 0.0)
                total_score = score + (sim_embedding * EMBEDDING_WEIGHT)

                if DEBUG_MODE:
                    debug_details.append({
                        "URL_404": row_404["URL_404"],
                        "URL_200_candidate": row_200["URL_200"],
                        "Fuzzy_score": fuzzy_score,
                        "Score_avant_embedding": score,
                        "Similarité_embedding": sim_embedding,
                        "Score_final": total_score,
                    })

                detailed_scored.append((row_200["URL_200"], total_score))

            filtered_scored = [m for m in detailed_scored if m[1] >= MIN_SCORE_THRESHOLD]
            filtered_scored.sort(key=lambda x: x[1], reverse=True)

            top_match_1 = filtered_scored[0] if len(filtered_scored) > 0 else (None, 0)
            top_match_2 = filtered_scored[1] if len(filtered_scored) > 1 else (None, 0)

        # Exact_ref + Parent
        ref_200_top1 = ""
        if top_match_1[0] is not None:
            candidate_row = df200[df200["URL_200"] == top_match_1[0]]
            if not candidate_row.empty:
                ref_200_top1 = candidate_row.iloc[0]["ref_200"]

        exact_ref = (
            "OUI"
            if ref_404 and ref_200_top1 and ref_404.strip().lower() == ref_200_top1.strip().lower()
            else "NON"
        )

        # Parent
        if PARENT_URL_LIST:
            if (PARENT_MATRIX is not None) and (embedding_404 is not None):
                v = embedding_404.astype(np.float32)
                v /= (np.linalg.norm(v) + 1e-9)
                sims = PARENT_MATRIX @ v

                Kp = 50
                k = min(Kp, sims.shape[0], len(PARENT_TOKENS), len(PARENT_URL_LIST))
                if k > 0:
                    idx = np.argpartition(sims, -k)[-k:]
                    idx = idx[np.argsort(sims[idx])[::-1]]

                    best_parent = None
                    best_score_parent = -1e18
                    for j in idx:
                        if (j < 0) or (j >= len(PARENT_TOKENS)) or (j >= len(PARENT_URL_LIST)):
                            continue
                        score_tokens = (
                            exact_token_bonus(tokens_404, PARENT_TOKENS[j])
                            + extra_tokens_malus(tokens_404, PARENT_TOKENS[j])
                        )
                        total_parent = score_tokens + float(sims[j]) * EMBEDDING_WEIGHT
                        if total_parent > best_score_parent:
                            best_score_parent = total_parent
                            best_parent = PARENT_URL_LIST[j]

                    parent_404 = best_parent or get_parent_url(top_match_1[0] or row_404["URL_404"])
                else:
                    parent_404 = get_parent_url(top_match_1[0] or row_404["URL_404"])
            else:
                parent_404 = get_parent_url(top_match_1[0]) if top_match_1[0] else get_parent_url(row_404["URL_404"])
        else:
            parent_404 = get_parent_url(top_match_1[0]) if top_match_1[0] else get_parent_url(row_404["URL_404"])

        best_matches.append(
            {
                "URL_404": row_404["URL_404"],
                "URL_200_top1": top_match_1[0],
                "similarity_top1": top_match_1[1],
                "PARENT_URL_404": parent_404,
                "EXACT_REF": exact_ref,
            }
        )

        # Progression
        percent_done = int(((i + 1) / total_404) * 100)
        _progress(30 + int(60 * (i + 1) / total_404), f"Progression : {percent_done}%")

    tick(times, "t_loop_end")

    # Export résultat
    tick(times, "t_export_start")
    df_result = pd.DataFrame(best_matches)

    def color_same_ref(row):
        if row["EXACT_REF"] == "OUI":
            return ["background-color: red"] * len(row)
        else:
            return [""] * len(row)

    df_result_styled = df_result.style.apply(color_same_ref, axis=1)
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df_result_styled.to_excel(writer, index=False)
    except Exception as e:
        raise RuntimeError(f"Impossible d'écrire le fichier résultat : {e}") from e

    tick(times, "t_export_end")
    _progress(95, "Export des résultats terminé")

    # Debug DF
    debug_df = None
    if DEBUG_MODE and len(debug_details) > 0:
        debug_df = pd.DataFrame(debug_details)

    # Histogramme
    tick(times, "t_hist_start")
    if create_hist:
        plt.figure()
        plt.hist(df_result["similarity_top1"], bins=20, alpha=0.7)
        plt.xlabel("Score de Similarité (top1) (Fuzzy + Sémantique)")
        plt.ylabel("Nombre d'URLs")
        plt.title(f"Distribution des Scores (Min = {MIN_SCORE_THRESHOLD})")
        # NOTE: en contexte serveur, on préférera sauvegarder la figure plutôt que plt.show()
        # plt.show()
    tick(times, "t_hist_end")

    tick(times, "end_all")

    total_time = elapsed(times, "start_all", "end_all")
    _log("=== Temps d'exécution par étape ===")
    _log(f"Chargement données : {elapsed(times, 't_load_start', 't_load_end'):.2f}s")
    _log(f"Caractéristiques dynamiques : {elapsed(times, 't_dyn_start', 't_dyn_end'):.2f}s")
    if EMBEDDING_MODEL is not None:
        _log(f"Embeddings : {elapsed(times, 't_emb_start', 't_emb_end'):.2f}s")
    _log(f"Substitution synonymes : {elapsed(times, 't_subs_start', 't_subs_end'):.2f}s")
    _log(f"Boucle matching : {elapsed(times, 't_loop_start', 't_loop_end'):.2f}s")
    _log(f"Export résultats : {elapsed(times, 't_export_start', 't_export_end'):.2f}s")
    _log(f"Histogramme : {elapsed(times, 't_hist_start', 't_hist_end'):.2f}s")
    _log(f"TOTAL : {total_time:.2f}s")

    _progress(100, "Matching terminé")

    return df_result, debug_df, output_file, logs, times
