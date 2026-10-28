import os
import sys
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.ner_model import get_ner_pipeline, detect_language


def match_entity(pred: Dict[str, Any], gold: Dict[str, Any], mode: str = "exact") -> bool:
    if mode == "exact":
        return pred["start"] == gold["start"] and pred["end"] == gold["end"] and pred["label"] == gold["label"]
    elif mode == "partial":
        overlap = max(0, min(pred["end"], gold["end"]) - max(pred["start"], gold["start"]))
        return overlap > 0 and pred["label"] == gold["label"]
    elif mode == "type_only":
        return pred["label"] == gold["label"]
    return False


def evaluate_predictions(gold_list: List[Dict[str, Any]], pred_list: List[Dict[str, Any]], mode: str = "exact"):
    matched_gold = set()
    matched_pred = set()

    tp = 0
    fps = []


    for p_idx, pred in enumerate(pred_list):
        match_found = False

        for g_idx, gold in enumerate(gold_list):
            if g_idx not in matched_gold and match_entity(pred, gold, mode=mode):
                matched_gold.add(g_idx)
                matched_pred.add(p_idx)
                tp += 1
                match_found = True
                break

        if not match_found:
            fps.append(pred)


    fns = [gold for g_idx, gold in enumerate(gold_list) if g_idx not in matched_gold]
    fp = len(fps)
    fn = len(fns)

    return tp, fp, fn, fps, fns


def run_evaluation(data_path: str = "data/evaluation_examples.json", mode: str = "exact") -> Dict[str, Any]:
    if not os.path.exists(data_path):
        fallback_path = os.path.join(os.path.dirname(__file__), "..", data_path)
        if os.path.exists(fallback_path):
            data_path = fallback_path


    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    pipeline = get_ner_pipeline()
    results = {}


    for model_type in ["baseline", "adapted"]:
        total_tp = 0
        total_fp = 0
        total_fn = 0


        per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        per_lang = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "docs": 0})
        error_log = []


        for ex in examples:
            combined_text = ex["combined_text"]
            gold_entities = ex["entities"]
            lang = detect_language(combined_text)


            pred_entities = pipeline.extract_entities(combined_text, model_type=model_type)


            tp, fp, fn, fps, fns = evaluate_predictions(gold_entities, pred_entities, mode=mode)
            total_tp += tp
            total_fp += fp
            total_fn += fn


            per_lang[lang]["tp"] += tp
            per_lang[lang]["fp"] += fp
            per_lang[lang]["fn"] += fn
            per_lang[lang]["docs"] += 1


            for p in pred_entities:
                is_tp = any(match_entity(p, g, mode=mode) for g in gold_entities)
                if is_tp:
                    per_label[p["label"]]["tp"] += 1
                else:
                    per_label[p["label"]]["fp"] += 1

            for g in gold_entities:
                is_tp = any(match_entity(p, g, mode=mode) for p in pred_entities)
                if not is_tp:
                    per_label[g["label"]]["fn"] += 1


            if fps or fns:
                error_log.append({
                    "doc_id": ex["id"],
                    "language": lang,
                    "false_positives": fps,
                    "false_negatives": fns
                })


        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


        label_metrics = {}
        for lbl, counts in per_label.items():
            l_tp = counts["tp"]
            l_fp = counts["fp"]
            l_fn = counts["fn"]
            l_prec = l_tp / (l_tp + l_fp) if (l_tp + l_fp) > 0 else 0.0
            l_rec = l_tp / (l_tp + l_fn) if (l_tp + l_fn) > 0 else 0.0
            l_f1 = (2 * l_prec * l_rec) / (l_prec + l_rec) if (l_prec + l_rec) > 0 else 0.0
            label_metrics[lbl] = {
                "precision": round(l_prec, 4),
                "recall": round(l_rec, 4),
                "f1": round(l_f1, 4),
                "support": l_tp + l_fn
            }


        lang_metrics = {}
        for lng, counts in per_lang.items():
            lg_tp = counts["tp"]
            lg_fp = counts["fp"]
            lg_fn = counts["fn"]
            lg_prec = lg_tp / (lg_tp + lg_fp) if (lg_tp + lg_fp) > 0 else 0.0
            lg_rec = lg_tp / (lg_tp + lg_fn) if (lg_tp + lg_fn) > 0 else 0.0
            lg_f1 = (2 * lg_prec * lg_rec) / (lg_prec + lg_rec) if (lg_prec + lg_rec) > 0 else 0.0
            lang_metrics[lng] = {
                "precision": round(lg_prec, 4),
                "recall": round(lg_rec, 4),
                "f1": round(lg_f1, 4),
                "docs": counts["docs"]
            }


        results[model_type] = {
            "overall": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn
            },
            "per_label": label_metrics,
            "per_language": lang_metrics,
            "errors": error_log
        }

    return results


if __name__ == "__main__":
    res = run_evaluation()


    print("===================================================================")
    print("       MODEL COMPARISON: BASELINE vs ADAPTED (EXACT MATCH)         ")
    print("===================================================================")
    print(f"BASELINE: Precision={res['baseline']['overall']['precision']:.4f}, Recall={res['baseline']['overall']['recall']:.4f}, F1={res['baseline']['overall']['f1']:.4f}")
    print(f"ADAPTED:  Precision={res['adapted']['overall']['precision']:.4f}, Recall={res['adapted']['overall']['recall']:.4f}, F1={res['adapted']['overall']['f1']:.4f}")


    print("\n-------------------------------------------------------------------")
    print(" MULTILINGUAL PERFORMANCE COMPARISON (ENGLISH vs MALAY/REGIONAL)")
    print("-------------------------------------------------------------------")
    for lang, m in res['adapted']['per_language'].items():
        lang_name = "English (en)" if lang == "en" else "Malay / Regional (ms)"
        print(f"  {lang_name:25s} -> P: {m['precision']:.2f}, R: {m['recall']:.2f}, F1: {m['f1']:.2f} ({m['docs']} docs)")


    print("\n-------------------------------------------------------------------")
    print(" PER-LABEL METRICS (ADAPTED MODEL)")
    print("-------------------------------------------------------------------")
    for lbl, m in res['adapted']['per_label'].items():
        print(f"  {lbl:10s} -> P: {m['precision']:.2f}, R: {m['recall']:.2f}, F1: {m['f1']:.2f} (support: {m['support']})")


    print("\n-------------------------------------------------------------------")
    print(" SAMPLE ERRORS FOR DISCUSSION (FIRST 3)")
    print("-------------------------------------------------------------------")
    errs = res['adapted']['errors'][:3]
    for e in errs:
        print(f"Doc: {e['doc_id']} [Lang: {e.get('language', 'en')}]")
        if e['false_positives']:
            print("  False Positives:", e['false_positives'])
        if e['false_negatives']:
            print("  False Negatives:", e['false_negatives'])
