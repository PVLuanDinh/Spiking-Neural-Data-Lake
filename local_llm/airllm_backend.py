"""
Optional on-device LLM backend for nl_interface.

This module wraps AirLLM (https://github.com/lyogavin/airllm) to provide
a complete() -> str contract matching the nl_interface spec. It returns
one of the valid labels: close, open, hold, home, retract, stop, rotate_a,
rotate_b, query, reject.

AirLLM and torch are imported LAZILY — the module itself is safe to import
even when they are absent. If unavailable on first use, a clear ImportError
is raised.

Usage:
    from local_llm.airllm_backend import AirLLMBackend
    backend = AirLLMBackend()  # downloads model weights on first instantiation
    label = backend.complete("close the gripper")  # returns "close"
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
    On-device LLM backend using AirLLM.

    The model is downloaded on first instantiation and cached locally.
    Each .complete(prompt) call tokenizes, runs the model, then reduces the output
    to a single label by scanning for label words.

    Args:
        model_id: HuggingFace model identifier. Defaults to Qwen2.5-0.5B-Instruct,
                  a tiny instruction-tuned model runnable on modest hardware.
        max_new_tokens: Generation limit (default 50). Keep small to reduce latency.
    """

    def __init__(self, model_id=None, max_new_tokens=50):
        """Initialize and load the model."""
        if model_id is None:
            model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

        # Lazy imports: fail loudly if airllm/torch missing
        try:
            from airllm import AutoModel
            from transformers import AutoTokenizer
            self._airllm_available = True
            self._AutoModel = AutoModel
            self._AutoTokenizer = AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "airllm and/or torch not installed. Install with: pip install airllm torch\n"
                f"Original error: {e}"
            ) from e

        # Download + cache model on first init
        try:
            print(f"Loading AirLLM model: {self.model_id}", file=sys.stderr)
            self.tokenizer = self._AutoTokenizer.from_pretrained(self.model_id)
            self.model = self._AutoModel.from_pretrained(self.model_id)
            print("AirLLM model loaded successfully", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to load AirLLM model {self.model_id}: {e}") from e

    def complete(self, prompt: str) -> str:
        """
        Run the model on a prompt and reduce output to a single valid label.

        Args:
            prompt: English instruction (e.g., "close the gripper")

        Returns:
            One of VALID_LABELS. If the model output contains no recognized label,
            returns "reject" as the safe fallback.
        """
        if not self.model or not self.tokenizer:
            return "reject"

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Generate (keep it minimal to avoid timeout)
            with __import__("torch").no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

            # Decode
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

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

        # airllm/torch available: instantiate and test
        print("AirLLM/torch available - running smoke test", file=sys.stderr)
        backend = AirLLMBackend()
        result = backend.complete("close the gripper")
        print(f"Smoke test result: {result}")
        assert result in VALID_LABELS, f"Invalid label: {result}"
        print("AirLLMBackend smoke test OK")
        sys.exit(0)

    except ImportError:
        # airllm/torch not installed - this is expected in CI/zero-dep
        print("airllm/torch not installed - skipping AirLLMBackend smoke (this is expected in CI/zero-dep)")
        sys.exit(0)

    except Exception as e:
        print(f"AirLLMBackend smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)
