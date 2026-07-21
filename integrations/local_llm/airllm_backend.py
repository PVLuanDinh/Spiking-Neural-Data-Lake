"""
Optional on-device LLM backend for nl_interface -- and the general-purpose local
inference entry point for this repo (see local_llm/run_local.py's `infer` subcommand).

This module wraps AirLLM (https://github.com/lyogavin/airllm), an INFERENCE-ONLY
library: it streams a large transformer's layers from disk one at a time -- keeping
only a single layer resident in memory at once -- so a 70B-671B parameter model can
run on a machine with only a few GB of VRAM, at the cost of per-token disk I/O
latency. AirLLM has no training or gradient path, and neither does this module; the
only thing this backend ever does is call .generate() in inference mode (no_grad,
greedy decoding).

Two usage shapes are exposed:

  - .generate(prompt) -> str     General-purpose inference: raw prompt in, raw
                                  generated text out. This is the first-class local
                                  large-LLM path -- call it directly, or run
                                  `python local_llm/run_local.py infer "<prompt>"`.

  - .complete(prompt) -> str     The nl_interface contract: same generation, reduced
                                  to one of VALID_LABELS (close, open, hold, ...).
                                  Used by the NL command router; never returns raw
                                  model text, and defaults to "reject" on any
                                  inference error or unrecognized output.

AirLLM and torch are imported LAZILY, inside __init__ / generate() -- this module is
always safe to import even when they are absent (run_local.py imports this module
unconditionally; AirLLM/torch are only touched once the `infer` subcommand or the
AirLLMBackend constructor actually runs). If the dependency is missing at that point,
a plain ImportError is raised with the one-line fix: pip install ".[llm]".

Usage:
    from local_llm.airllm_backend import AirLLMBackend
    backend = AirLLMBackend()  # downloads model weights on first instantiation
    label = backend.complete("close the effector")   # -> "close" (nl_interface contract)
    text = backend.generate("Explain what a spike train is in one sentence.")  # inference
"""

import sys
from pathlib import Path

# Shim: allow imports from src/ when this file is run as __main__
repo_root = Path(__file__).parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

# Label list: the only valid outputs
VALID_LABELS = {"close", "open", "hold", "home", "retract", "stop", "rotate_a", "rotate_b", "query", "reject"}


class AirLLMBackend:
    """
    On-device LLM backend using AirLLM's disk-streaming inference engine.

    AirLLM keeps a large model's weights on disk and loads one transformer layer at a
    time into memory to run the forward pass, then frees it before loading the next --
    this is what lets a 70B+ model run without 70B+ of VRAM. It is pure inference:
    there is no gradient computation or weight update anywhere in this class. The
    model is downloaded (and, on first use, decomposed into per-layer shards) on
    first instantiation, then cached locally for offline reuse.

    Args:
        model_id: HuggingFace model identifier. Defaults to Qwen2.5-0.5B-Instruct, a
                  tiny instruction-tuned model runnable on modest hardware. Pass a
                  much larger id (e.g. a 70B-class model) to exercise AirLLM's actual
                  reason for existing: layer-by-layer disk streaming into a low-VRAM
                  budget.
        max_new_tokens: Generation limit (default 50). Keep small to reduce latency --
                  every generated token re-streams every layer from disk.
        compression: Optional block-wise weight compression forwarded to AirLLM's
                  AutoModel.from_pretrained -- "4bit" or "8bit" (requires
                  bitsandbytes). Speeds up the disk-load bottleneck at some accuracy
                  cost. Default None (no compression, full-precision layer shards).
    """

    def __init__(self, model_id=None, max_new_tokens=50, compression=None):
        """Initialize and load the model (inference only -- no training path)."""
        if model_id is None:
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.compression = compression
        self.model = None
        self.tokenizer = None

        # Lazy imports: fail loudly (but cleanly, one line) if airllm/torch missing
        try:
            from airllm import AutoModel
            from transformers import AutoTokenizer
            self._airllm_available = True
            self._AutoModel = AutoModel
            self._AutoTokenizer = AutoTokenizer
        except ImportError as e:
            raise ImportError(
                'airllm and/or torch not installed. Install with: pip install ".[llm]"\n'
                f"Original error: {e}"
            ) from e

        # Download + cache model on first init. This is where AirLLM decomposes the
        # model into per-layer shards on disk (first run only) -- the actual
        # disk-streaming inference engine is constructed here, not trained.
        try:
            print(f"Loading AirLLM model: {self.model_id}", file=sys.stderr)
            self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_id)
            if self.compression is not None:
                self.model = self._AutoModel.from_pretrained(self.model_id, compression=self.compression)
            else:
                self.model = self._AutoModel.from_pretrained(self.model_id)
            print("AirLLM model loaded successfully", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to load AirLLM model {self.model_id}: {e}") from e

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        """
        General-purpose inference: run the model on a prompt and return the raw
        decoded text -- no label reduction. This is the first-class local large-LLM
        entry point (see local_llm/run_local.py's `infer` subcommand). Every call
        re-streams the model's layers from disk one at a time (AirLLM's disk-
        streaming engine); there is no training or gradient computation anywhere in
        this path -- generation runs under torch.no_grad() with greedy decoding.

        Args:
            prompt: Free-form text prompt.
            max_new_tokens: Overrides the instance default for this call only.

        Returns:
            The decoded generated text (special tokens stripped).

        Raises:
            RuntimeError: if the model/tokenizer failed to load.
            Exception: generation-time errors propagate uncaught here -- callers that
                       want a safe fallback (like .complete()) catch them themselves.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("AirLLM model not loaded")

        import torch  # lazy: only needed once we actually generate

        tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():  # inference only -- no gradients, ever
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=tokens,
                do_sample=False,
                temperature=0.0,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def complete(self, prompt: str) -> str:
        """
        Run the model on a prompt and reduce output to a single valid label.

        This is the nl_interface contract (see module docstring): free text in, one
        of VALID_LABELS out, "reject" on any error or unrecognized output. For raw
        text generation, use .generate() instead.

        Args:
            prompt: English instruction (e.g., "close the effector")

        Returns:
            One of VALID_LABELS. If the model output contains no recognized label,
            or generation fails for any reason, returns "reject" as the safe fallback.
        """
        if not self.model or not self.tokenizer:
            return "reject"

        try:
            output_text = self.generate(prompt)

            # Scan for label words (case-insensitive)
            output_lower = output_text.lower()
            for label in VALID_LABELS:
                if label in output_lower:
                    return label

            # No recognized label found: reject safely
            return "reject"

        except Exception as e:
            # Model inference failed: reject safely and log
            print(f"AirLLM inference error: {e}", file=sys.stderr)
            return "reject"


if __name__ == "__main__":
    # Smoke test: if airllm/torch are unavailable, skip gracefully
    try:
        import torch  # noqa: F401
        from airllm import AutoModel  # noqa: F401

        # airllm/torch available: instantiate and test both entry points
        print("AirLLM/torch available - running smoke test", file=sys.stderr)
        backend = AirLLMBackend()

        result = backend.complete("close the effector")
        print(f"complete() smoke test result: {result}")
        assert result in VALID_LABELS, f"Invalid label: {result}"

        text = backend.generate("Say hello.", max_new_tokens=8)
        print(f"generate() smoke test result: {text!r}")
        assert isinstance(text, str) and len(text) > 0, "generate() returned empty output"

        print("AirLLMBackend smoke test OK")
        sys.exit(0)

    except ImportError:
        # airllm/torch not installed - this is expected in CI/zero-dep
        print("airllm/torch not installed - skipping AirLLMBackend smoke (this is expected in CI/zero-dep)")
        sys.exit(0)

    except Exception as e:
        print(f"AirLLMBackend smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)
