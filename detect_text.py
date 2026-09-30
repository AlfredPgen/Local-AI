r"""Was this text written by my model? Two tests.

1. Watermark (definitive, if you generated with a key):
       python tiny_gpt.py --generate "Genetic drift is" --watermark-key my-secret
       python detect_text.py sample.txt --watermark-key my-secret
   Watermarked generation nudges each token towards a pseudo-random "green"
   list derived from the key and the previous token. Without the key the list
   is invisible; with it, count green tokens: human text has about gamma
   (25%) green, watermarked text far more. Each distinct (previous token,
   token) pair is counted once: a repeated pair would repeat the same
   green/red outcome and inflate the evidence. Under "not watermarked" the
   green count is then binomial, and the p-value is its exact upper tail.

2. Likelihood test (evidence only, works without a watermark):
       python detect_text.py sample.txt --checkpoint tinyGPT_best.pt
   Text sampled from a model is, on average, exactly as predictable to that
   model as its own samples are. For each token we compare the observed
   log-probability with the mean and variance the model itself expects
   (the analytic criterion of Fast-DetectGPT, Bao et al. 2024):
       score = sum_t (log p(x_t) - mu_t) / sqrt(sum_t sigma_t^2)
   Around 0 or negative: nothing unusual. Clearly positive (> 2-3): the text
   is more predictable than this model's own random samples, as produced by
   sharpened sampling (temperature < 1, top-k, top-p) from this or a similar
   model. Text sampled at temperature 1 without truncation is NOT detectable
   this way. It cannot prove authorship: calibrate with --calibrate, which
   scores your own model samples and validation text.

Run with --device cpu while training uses the GPU.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tiny_gpt  # noqa: E402


def binomial_sf(green, t, gamma):
    """P(X >= green) for X ~ Binomial(t, gamma): the exact watermark p-value."""
    from scipy.stats import binom
    return float(binom.sf(green - 1, t, gamma)) if t else float("nan")


def watermark_z(ids, key, vocab_size, gamma):
    """(z, green, t): distinct (previous, token) pairs only, so repeated phrases,
    tables or boilerplate cannot inflate the count."""
    scored = sorted({(ids[i - 1], ids[i]) for i in range(1, len(ids))})
    if not scored:
        return float("nan"), 0, 0
    green = sum(bool(tiny_gpt.watermark_green(key, prev, vocab_size, gamma)[tok]) for prev, tok in scored)
    t = len(scored)
    return (green - gamma * t) / math.sqrt(t * gamma * (1 - gamma)), green, t


@torch.no_grad()
def likelihood_score(model, ids, device):
    """(analytic Fast-DetectGPT score, mean log-prob per token, perplexity)."""
    obs_sum = mu_sum = var_sum = 0.0
    n = 0
    for start in range(0, len(ids) - 1, model.ctx):
        chunk = ids[start:start + model.ctx + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], device=device)
        y = torch.tensor(chunk[1:], device=device)
        logp = F.log_softmax(model(x)[0].float(), dim=-1)
        p = logp.exp()
        obs = logp.gather(-1, y[:, None]).squeeze(-1)
        mu = (p * logp).sum(-1)
        var = (p * logp.pow(2)).sum(-1) - mu.pow(2)
        obs_sum += obs.sum().item()
        mu_sum += mu.sum().item()
        var_sum += var.clamp(min=0).sum().item()
        n += len(chunk) - 1
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return (obs_sum - mu_sum) / math.sqrt(max(var_sum, 1e-9)), obs_sum / n, math.exp(-obs_sum / n)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="a .txt/.md file, or the text itself in quotes")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "tinyGPT_best.pt"))
    p.add_argument("--watermark-key")
    p.add_argument("--watermark-gamma", type=float, default=0.25)
    p.add_argument("--calibrate", action="store_true",
                   help="score 5 samples from the model and 5 validation passages to show typical values")
    p.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    p.add_argument("--trust-checkpoint", action="store_true")
    args = p.parse_args()
    if not args.text and not args.calibrate:
        p.error("give a text/file or --calibrate")
    if not 0 < args.watermark_gamma < 1:
        p.error("--watermark-gamma must be between 0 and 1")
    device = tiny_gpt.select_device(args.device)
    model, tok, cfg, obj = tiny_gpt.load_for_inference(args.checkpoint, device, args.trust_checkpoint)

    def report(label, text):
        prefix = [tok.bos_id] if tok.bos_id >= 0 else []
        ids = prefix + tok.encode(text)
        score, mean_lp, ppl = likelihood_score(model, ids, device)
        line = f"{label}: {len(ids) - len(prefix)} tokens | likelihood score {score:+.2f} | perplexity {ppl:.1f}"
        if args.watermark_key:
            z, green, t = watermark_z(ids[len(prefix):], args.watermark_key, cfg.vocab_size, args.watermark_gamma)
            line += (f" | watermark z {z:+.2f} ({green}/{t} distinct pairs green, "
                     f"p = {binomial_sf(green, t, args.watermark_gamma):.1e})")
        print(line)
        return score

    if args.text:
        text = args.text
        if os.path.isfile(text):
            with open(text, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        score = report("text", text)
        if not math.isfinite(score):
            print("Likelihood test: too short to score.")
        else:
            verdict = ("more predictable than this model's own samples: typical of its sharpened sampling"
                       if score > 3 else "ambiguous" if score > 1
                       else "no sign of this model's sharpened sampling (pure sampling cannot be detected)")
            print(f"Likelihood test: {verdict} (evidence, not proof).")
        if args.watermark_key:
            print("Watermark test: p < 1e-4 means the text was generated with this key; a large p means no evidence.")
    if args.calibrate:
        print("\nCalibration: model samples (should score high) ...")
        gen = torch.Generator().manual_seed(0)
        wm = (args.watermark_key, args.watermark_gamma, 2.0) if args.watermark_key else None
        for prompt in ("Genetic drift is", "The heritability of", "In statistics,", "Proteins are", "A GWAS"):
            report(f"  sample '{prompt}'", tiny_gpt.generate(model, tok, prompt, 150, device, None, 0.8, 50, 0.95,
                                                             gen, wm))
        ds = (obj.get("dataset") or {}).get("path")
        if ds and os.path.isdir(ds):
            import compare_models
            print("... and validation text written by people (should score near 0):")
            for source, k, text in compare_models.validation_documents(ds, 3, 3000)[:5]:
                report(f"  {source} document {k}", text)


if __name__ == "__main__":
    main()
