from typing import List, Dict, Any, Optional
import re
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler


from src.domain_patterns import (
    LABEL_MAPPING,
    FINANCIAL_NEWS_PATTERNS,
    CHINESE_NEWS_PATTERNS,
    ENTITY_NORMALIZATIONS,
    DOMAIN_PATTERNS_LOWER,
    normalize_entity,
)


MALAY_COMMON_WORDS = {
    "bukan", "tapi", "tetapi", "peluang", "besar", "promosikan", "intipati", "berita",
    "pengunjung", "disasarkan", "hadir", "termasuk", "pelancong", "antarabangsa",
    "dijangka", "menjana", "impak", "ekonomi", "sekitar", "bilion", "juta", "kepada",
    "menyediakan", "pakej", "pelancongan", "untuk", "menggalakkan", "asing",
    "tinggal", "lebih", "lama", "turut", "menjadi", "platform", "budaya",
    "dan", "seni", "sempena", "kempen", "melawat", "kementerian", "menyasarkan",
    "kehadiran", "kira-kira", "penganjuran", "litar", "dari", "hingga", "menurut",
    "laporan", "daripada", "jumlah", "itu", "bagi", "menyaksikan", "perlumbaan",
    "berprestij", "tersebut", "ketua", "komunikasi", "korporat", "berkata",
    "berdasarkan", "unjuran", "data", "penyelidikan", "ini", "sahaja", "mampu",
    "pendapatan", "sebanyak", "melalui", "sektor", "sukan", "perbelanjaan",
    "setakat", "kita", "keseluruhan", "yang", "membanjiri", "ada", "sediakan",
    "bertepatan", "dengan", "memperkenalkan", "serta", "produk", "negara",
    "menerusi", "bersama", "beberapa", "luar", "melanjutkan", "tempoh",
    "penginapan", "mereka", "selain", "berbelanja", "membeli-belah", "melakukan",
    "aktiviti", "seperti", "destinasi", "menarik", "kerana", "selepas",
    "saingan", "seterusnya", "akan", "berlangsung", "siapa", "terlibat",
    "persiapan", "bawah", "jawatankuasa", "kini", "dilaporkan", "mencapai",
    "peratus", "pelbagai", "agensi", "memastikan", "berjalan", "lancar",
    "antaranya", "ialah", "hari", "kejadian", "berskala", "diatur",
    "maksudnya", "anda", "memeriahkan", "lagi", "ke", "di", "pada", "oleh",
    "adalah", "telah", "iaitu", "melancarkan", "bermula", "bantuan", "negeri"
}


def detect_language(text: str) -> str:
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"


    lower = text.lower()
    words = set(re.findall(r'\b[a-z-]+\b', lower))


    match_count = len(words.intersection(MALAY_COMMON_WORDS))
    if match_count >= 2:
        return "ms"

    return "en"


def is_malay_false_positive(ent_text: str, label: str) -> bool:
    clean = ent_text.strip().lower()
    words = [w for w in re.findall(r'\b[\w-]+\b', clean)]
    if not words:
        return True


    if len(words) == 1 and words[0] in MALAY_COMMON_WORDS:
        return True


    if label in ["PERSON", "GPE", "NORP", "PRODUCT", "ORG"]:
        if all(w in MALAY_COMMON_WORDS for w in words):
            return True

        if words[0] in {"tapi", "tetapi", "dan", "atau", "daripada", "serta", "seperti", "kerana", "ke", "di", "pada", "oleh", "selain", "menjana", "berlangsung"}:
            return True

    return False


class NERPipeline:
    def __init__(self, spacy_model_name: str = "en_core_web_sm"):
        self.spacy_model_name = spacy_model_name


        try:
            self.baseline_nlp = spacy.load(spacy_model_name)
        except OSError:
            try:
                from spacy.cli import download
                download(spacy_model_name)
                self.baseline_nlp = spacy.load(spacy_model_name)
            except Exception:
                self.baseline_nlp = spacy.load("en_core_web_sm")
                self.spacy_model_name = "en_core_web_sm"


        try:
            self.adapted_nlp = spacy.load(self.spacy_model_name)
        except Exception:
            self.adapted_nlp = spacy.load("en_core_web_sm")


        ruler = self.adapted_nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(FINANCIAL_NEWS_PATTERNS)


        self.zh_nlp = None
        self.baseline_zh_nlp = None
        self._init_chinese_pipeline()

    def _init_chinese_pipeline(self):
        try:
            self.baseline_zh_nlp = spacy.load("zh_core_web_sm")


            self.zh_nlp = spacy.load("zh_core_web_sm")
            zh_ruler = self.zh_nlp.add_pipe("entity_ruler", before="ner")
            zh_ruler.add_patterns(CHINESE_NEWS_PATTERNS)
        except Exception:
            self.zh_nlp = None
            self.baseline_zh_nlp = None

    def detect_language(self, text: str) -> str:
        return detect_language(text)

    def extract_entities(self, text: str, model_type: str = "adapted") -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        lang = self.detect_language(text)


        if lang == "zh":
            if model_type == "baseline" and self.baseline_zh_nlp is not None:
                return self._extract_chinese_baseline_entities(text)
            elif self.zh_nlp is not None:
                return self._extract_chinese_entities(text)


        nlp = self.adapted_nlp if model_type in ["adapted", "multilingual", "transformer"] else self.baseline_nlp


        doc = nlp(text)

        entities = []
        seen_spans = set()

        for ent in doc.ents:
            if ent.label_ not in LABEL_MAPPING:
                continue
            mapped_label = LABEL_MAPPING[ent.label_]

            ent_text = ent.text
            start_char = ent.start_char
            end_char = ent.end_char


            text_lower = ent_text.strip().lower()
            is_domain_match = (text_lower in DOMAIN_PATTERNS_LOWER or text_lower in ENTITY_NORMALIZATIONS)


            if not is_domain_match:
                prep_match = re.match(r'^(?:ke|di|pada|dari|oleh|bagi|dan|serta)\s+(.+)$', ent_text, re.IGNORECASE)
                if prep_match:
                    trimmed = prep_match.group(1).strip()
                    offset_shift = len(ent_text) - len(trimmed)
                    ent_text = trimmed
                    start_char += offset_shift
                    text_lower = ent_text.strip().lower()
                    is_domain_match = (text_lower in DOMAIN_PATTERNS_LOWER or text_lower in ENTITY_NORMALIZATIONS)


            if not is_domain_match and (lang == "ms" or model_type == "adapted"):
                if is_malay_false_positive(ent_text, mapped_label):
                    continue


            span_key = (start_char, end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)


            text_lower = ent.text.strip().lower()
            if text_lower in DOMAIN_PATTERNS_LOWER or text_lower in ENTITY_NORMALIZATIONS:
                confidence = 0.98
            else:
                has_vector = any(token.has_vector for token in ent)
                confidence = 0.92 if has_vector else 0.82


            normalized = normalize_entity(ent_text, mapped_label)


            entities.append({
                "text": ent_text,
                "label": mapped_label,
                "start": start_char,
                "end": end_char,
                "confidence": round(confidence, 2),
                "normalized": normalized
            })


        entities.sort(key=lambda x: (x["start"], x["end"]))
        return entities

    def _extract_chinese_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.zh_nlp(text)
        entities = []
        seen_spans = set()

        for ent in doc.ents:
            if ent.label_ not in LABEL_MAPPING:
                continue
            mapped_label = LABEL_MAPPING[ent.label_]

            ent_text = ent.text
            end_char = ent.end_char


            if ent_text.startswith("MHL") and ent_text != "MHL":
                ent_text = "MHL"
                end_char = ent.start_char + 3
            elif ent_text.startswith("MHC") and ent_text != "MHC":
                ent_text = "MHC"
                end_char = ent.start_char + 3

            span_key = (ent.start_char, end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            text_lower = ent_text.strip().lower()
            confidence = 0.98 if (text_lower in DOMAIN_PATTERNS_LOWER or text_lower in ENTITY_NORMALIZATIONS) else 0.89
            normalized = normalize_entity(ent_text, mapped_label)

            entities.append({
                "text": ent_text,
                "label": mapped_label,
                "start": ent.start_char,
                "end": end_char,
                "confidence": round(confidence, 2),
                "normalized": normalized
            })


        for m in re.finditer(r'\b(MHC|MHL|BNM|MDV|CIMB|RHB|Maybank)\b', text):
            span_key = (m.start(), m.end())

            if not any(e["start"] <= m.start() and m.end() <= e["end"] for e in entities):
                entities.append({
                    "text": m.group(),
                    "label": "ORG",
                    "start": m.start(),
                    "end": m.end(),
                    "confidence": 0.98,
                    "normalized": normalize_entity(m.group(), "ORG")
                })


        entities.sort(key=lambda x: (x["start"], x["end"]))
        return entities

    def _extract_chinese_baseline_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.baseline_zh_nlp(text)
        entities = []
        seen_spans = set()

        for ent in doc.ents:
            if ent.label_ not in LABEL_MAPPING:
                continue
            mapped_label = LABEL_MAPPING[ent.label_]

            span_key = (ent.start_char, ent.end_char)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            has_vector = any(token.has_vector for token in ent)
            confidence = 0.90 if has_vector else 0.85

            entities.append({
                "text": ent.text,
                "label": mapped_label,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": round(confidence, 2),
                "normalized": None
            })

        entities.sort(key=lambda x: (x["start"], x["end"]))
        return entities


_pipeline_instance: Optional[NERPipeline] = None

def get_ner_pipeline() -> NERPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = NERPipeline()
    return _pipeline_instance
