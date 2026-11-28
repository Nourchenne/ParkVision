"""Rapport de monitoring : agrège le journal JSONL des prédictions et applique les règles d'alerte.

    python src/analysis/monitor_report.py journal.jsonl [autre.jsonl ...] [--plot courbe.png]

Pour chaque journal : nombre de requêtes, taux d'erreur, latence médiane et p95, confiance
moyenne, part de requêtes sans aucune détection, puis les deux règles d'alerte (voir
docs/monitoring.md). Code de sortie 2 si une règle s'est déclenchée, 0 sinon.
Lecture seule : le journal n'est jamais modifié.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Fenêtre glissante : les 20 dernières prédictions réussies.
WINDOW = 20
# Règle 1 : confiance moyenne sous ce seuil sur la fenêtre.
CONFIDENCE_THRESHOLD = 0.5
# Règle 2 : part de requêtes sans détection au-dessus de ce seuil sur la fenêtre.
EMPTY_SHARE_THRESHOLD = 0.5

RULE_CONFIDENCE = "confiance_basse"
RULE_EMPTY = "resultats_vides"


def load(path: Path) -> list[dict]:
    """Lit le journal, précédé de sa rotation `<nom>.1` si elle existe. Ignore les lignes illisibles."""
    records: list[dict] = []
    for candidate in (path.with_name(path.name + ".1"), path):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error")]
    latencies = [r["inference_ms"] for r in ok if r.get("inference_ms") is not None]
    confidences = [r["conf_mean"] for r in ok if r.get("conf_mean") is not None]
    totals = [r["total"] for r in ok if r.get("total") is not None]
    return {
        "requetes": len(records),
        "erreurs": len(records) - len(ok),
        "taux_erreur": (len(records) - len(ok)) / len(records) if records else 0.0,
        "latence_mediane_ms": statistics.median(latencies) if latencies else None,
        "latence_p95_ms": percentile(latencies, 0.95) if latencies else None,
        "confiance_moyenne": statistics.fmean(confidences) if confidences else None,
        "part_sans_detection": (sum(1 for r in ok if r.get("total") == 0) / len(ok)) if ok else None,
        "detections_moyennes": statistics.fmean(totals) if totals else None,
    }


def check_window(ok_window: list[dict], confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 empty_threshold: float = EMPTY_SHARE_THRESHOLD) -> list[str]:
    """Règles d'alerte sur une fenêtre de prédictions réussies ; renvoie les règles déclenchées."""
    fired = []
    confidences = [r["conf_mean"] for r in ok_window if r.get("conf_mean") is not None]
    if confidences and statistics.fmean(confidences) < confidence_threshold:
        fired.append(RULE_CONFIDENCE)
    empty = sum(1 for r in ok_window if r.get("total") == 0) / len(ok_window)
    if empty > empty_threshold:
        fired.append(RULE_EMPTY)
    return fired


def rolling_alerts(records: list[dict], window: int = WINDOW, **thresholds) -> list[tuple[int, list[str]]]:
    """Évalue les règles après chaque prédiction réussie dès que la fenêtre est pleine.

    Renvoie (rang de la prédiction, règles déclenchées) pour chaque position en alerte.
    """
    ok = [r for r in records if not r.get("error")]
    alerts = []
    for index in range(window - 1, len(ok)):
        fired = check_window(ok[index - window + 1:index + 1], **thresholds)
        if fired:
            alerts.append((index + 1, fired))
    return alerts


def format_report(label: str, records: list[dict], window: int, **thresholds) -> tuple[str, bool]:
    s = summarize(records)

    def fmt(value, pattern):
        return "n/a" if value is None else pattern.format(value)

    lines = [f"== {label} ==",
             f"requêtes : {s['requetes']}   erreurs : {s['erreurs']} ({s['taux_erreur']:.1%})",
             f"latence : médiane {fmt(s['latence_mediane_ms'], '{:.1f}')} ms, p95 {fmt(s['latence_p95_ms'], '{:.1f}')} ms",
             f"confiance moyenne : {fmt(s['confiance_moyenne'], '{:.3f}')}",
             f"détections par requête (moyenne) : {fmt(s['detections_moyennes'], '{:.1f}')}",
             f"part de requêtes sans détection : {fmt(s['part_sans_detection'], '{:.1%}')}"]
    ok_count = s["requetes"] - s["erreurs"]
    alerts = rolling_alerts(records, window, **thresholds)
    if ok_count < window:
        lines.append(f"règles d'alerte : non évaluables ({ok_count} prédictions, il en faut {window})")
    elif alerts:
        first_index, first_rules = alerts[0]
        lines.append(f"ALERTE ({', '.join(first_rules)}) dès la prédiction n°{first_index} ; "
                     f"{len(alerts)} positions sur {ok_count - window + 1} en alerte.")
    else:
        lines.append(f"aucune alerte sur {ok_count - window + 1} fenêtres de {window} prédictions.")
    return "\n".join(lines), bool(alerts)


def plot(logs: list[tuple[str, list[dict]]], output: Path, window: int,
         confidence_threshold: float = CONFIDENCE_THRESHOLD, **thresholds) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib est requis pour --plot (pip install matplotlib).")

    figure, axes = plt.subplots(2, len(logs), figsize=(6.2 * len(logs), 6.4), squeeze=False,
                                sharey="row")
    for column, (label, records) in enumerate(logs):
        ok = [r for r in records if not r.get("error")]
        x = list(range(1, len(ok) + 1))
        top, bottom = axes[0][column], axes[1][column]
        mean_points = [(i, r["conf_mean"]) for i, r in zip(x, ok, strict=True) if r.get("conf_mean") is not None]
        min_points = [(i, r["conf_min"]) for i, r in zip(x, ok, strict=True) if r.get("conf_min") is not None]
        if mean_points:
            top.plot(*zip(*mean_points, strict=True), color="#1f6feb", marker="o", ms=3, lw=1.2,
                     label="confiance moyenne")
        if min_points:
            top.plot(*zip(*min_points, strict=True), color="#9aa5b1", lw=1, label="confiance minimale")
        top.axhline(confidence_threshold, color="#d1242f", ls="--", lw=1, label=f"seuil {confidence_threshold:g}")
        empty_x = [i for i, r in zip(x, ok, strict=True) if r.get("total") == 0]
        if empty_x:
            top.plot(empty_x, [0.02] * len(empty_x), "x", color="#d1242f", label="aucune détection")
        alerts = rolling_alerts(records, window, confidence_threshold=confidence_threshold, **thresholds)
        if alerts:
            top.axvspan(alerts[0][0], len(ok), color="#d1242f", alpha=0.08, label="alerte active")
        top.set_title(label, fontsize=11)
        top.set_ylim(0, 1.02)
        top.set_ylabel("confiance de détection")
        top.legend(loc="lower right" if column == 0 else "upper left", fontsize=8)
        bottom.bar(x, [r.get("total") or 0 for r in ok], color="#2da44e", width=0.9)
        bottom.set_xlabel("prédiction n°")
        bottom.set_ylabel("détections par image")
    figure.suptitle("Monitoring des prédictions : confiance et détections", fontsize=12)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="journaux JSONL à analyser")
    parser.add_argument("--label", action="append", help="nom de chaque journal (dans l'ordre)")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--empty-threshold", type=float, default=EMPTY_SHARE_THRESHOLD)
    parser.add_argument("--plot", type=Path, help="écrit la courbe de confiance (PNG)")
    args = parser.parse_args()

    labels = args.label or []
    thresholds = {"confidence_threshold": args.confidence_threshold, "empty_threshold": args.empty_threshold}
    loaded = []
    for index, path in enumerate(args.logs):
        records = load(path)
        if not records:
            sys.exit(f"{path} : journal absent ou vide.")
        loaded.append((labels[index] if index < len(labels) else path.stem, records))

    any_alert = False
    for label, records in loaded:
        text, alert = format_report(label, records, args.window, **thresholds)
        print(text + "\n")
        any_alert = any_alert or alert
    if args.plot:
        plot(loaded, args.plot, args.window, **thresholds)
        print(f"courbe -> {args.plot}")
    sys.exit(2 if any_alert else 0)


if __name__ == "__main__":
    main()
