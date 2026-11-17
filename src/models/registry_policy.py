"""Pure, fail-closed ParkVision champion comparison policy."""

import math
from pathlib import Path

import yaml

METRICS = ("map50_95", "mae_empty", "mae_occupied", "exact_count_accuracy")
LIMITS = ("min_exact_gain", "max_map_drop", "max_mae_empty_increase",
          "max_mae_occupied_increase")


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a configured nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a configured nonnegative number")
    return float(value)


def load_policy(path):
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(policy, dict) or not isinstance(policy.get("promotion"), dict):
        raise TypeError("promotion policy must be a mapping")
    p = policy["promotion"]
    if set(p) != {"enabled", "lock_file", *LIMITS, "bootstrap", "latency"}:
        raise ValueError("promotion policy has missing or unknown fields")
    if not isinstance(p["enabled"], bool):
        raise TypeError("promotion.enabled must be boolean")
    if p["bootstrap"] not in ("manual", "allow_eligible_candidate"):
        raise ValueError("invalid bootstrap mode")
    latency = p["latency"]
    if not isinstance(latency, dict) or set(latency) != {
        "mode", "max_relative_increase", "min_samples", "min_warmup_runs"
    }:
        raise ValueError("invalid latency policy")
    if latency["mode"] != "require_if_recorded":
        raise ValueError("invalid latency mode")
    if p["enabled"]:
        if (not isinstance(p["lock_file"], str) or
            not Path(p["lock_file"]).is_absolute()):
            raise ValueError("enabled promotion requires an absolute shared lock_file")
        for key in LIMITS:
            nonnegative(p[key], key)
        nonnegative(latency["max_relative_increase"], "latency.max_relative_increase")
        for key in ("min_samples", "min_warmup_runs"):
            value = latency[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"latency.{key} must be a positive integer")
    return p


def compare(candidate, champion, policy):
    """Return a decision; malformed or incomparable evidence never promotes."""
    reasons = []
    if not policy["enabled"]:
        return {"promote": False, "reasons": ["automatic promotion disabled"]}
    for label, item in (("candidate", candidate), ("champion", champion)):
        if item is None:
            continue
        if item.get("quality_status") != "passed" or not item.get("verified"):
            reasons.append(f"{label} quality or provenance unverified")
        if not item.get("comparison_verified"):
            reasons.append(f"{label} evaluation environment unverified")
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            reasons.append(f"{label} metrics missing")
        else:
            for key in METRICS:
                try:
                    value = nonnegative(metrics.get(key), f"{label}.{key}")
                    if key in ("map50_95", "exact_count_accuracy") and value > 1:
                        raise ValueError("metric exceeds 1")
                except (ValueError, TypeError):
                    reasons.append(f"{label} {key} missing or invalid")
        if not item.get("protocol_id"):
            reasons.append(f"{label} evaluation protocol missing")
    if reasons:
        return {"promote": False, "reasons": reasons}
    if champion is None:
        allowed = policy["bootstrap"] == "allow_eligible_candidate"
        return {"promote": allowed, "reasons": ["explicit bootstrap policy" if allowed
                                                  else "manual bootstrap required"]}
    if candidate["protocol_id"] != champion["protocol_id"]:
        reasons.append("evaluation protocols differ")
    cm, hm = candidate["metrics"], champion["metrics"]
    gain = cm["exact_count_accuracy"] - hm["exact_count_accuracy"]
    if gain <= 0 or gain < policy["min_exact_gain"]:
        reasons.append("joint exact-count accuracy did not improve enough")
    if cm["map50_95"] < hm["map50_95"] - policy["max_map_drop"]:
        reasons.append("mAP50-95 regression exceeds limit")
    for field, limit in (("mae_empty", "max_mae_empty_increase"),
                         ("mae_occupied", "max_mae_occupied_increase")):
        if cm[field] > hm[field] + policy[limit]:
            reasons.append(f"{field} regression exceeds limit")
    cl, hl = candidate.get("latency"), champion.get("latency")
    if cl is None and hl is None:
        pass
    elif not isinstance(cl, dict) or not isinstance(hl, dict):
        reasons.append("latency evidence exists for only one model")
    elif (cl.get("protocol_id") != hl.get("protocol_id") or
          not cl.get("protocol_id")):
        reasons.append("latency protocols differ")
    else:
        try:
            c_ms = nonnegative(cl.get("milliseconds"), "candidate latency")
            h_ms = nonnegative(hl.get("milliseconds"), "champion latency")
            if any(item.get("sample_count", 0) < policy["latency"]["min_samples"] or
                   item.get("warmup_runs", 0) < policy["latency"]["min_warmup_runs"]
                   for item in (cl, hl)):
                reasons.append("latency benchmark has insufficient samples or warm-up")
            elif c_ms <= 0 or h_ms <= 0 or c_ms > h_ms * (
                1 + policy["latency"]["max_relative_increase"]
            ):
                reasons.append("latency regression exceeds limit")
        except (ValueError, TypeError):
            reasons.append("invalid latency evidence")
    return {"promote": not reasons, "reasons": reasons or ["policy passed"]}
