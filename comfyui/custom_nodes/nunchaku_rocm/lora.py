"""
LoRAs for the Nunchaku adapters. :class:`NunchakuLoraLoader` (nodes.py) only records ``(path, strength)`` in the
``MODEL``'s ``transformer_options``, so ``MODEL`` branches with and without LoRAs share one loaded model; each adapter
applies the requested set lazily, when it changes, with ``nunchaku.lora.runtime``.
"""

import comfy.utils
from torch import nn


class LoraMixin:
    """Adds :meth:`_sync_loras`; call :meth:`_init_loras` in ``__init__``."""

    def _init_loras(self) -> None:
        self._applied_loras: tuple = ()
        self._lora_cache: dict[str, dict] = {}

    def _lora_model(self) -> nn.Module:
        return self.transformer

    def _sync_loras(self, transformer_options: dict) -> None:
        requested = tuple(transformer_options.get("nunchaku_loras", ()))
        if requested != self._applied_loras:
            self._apply_loras(requested)

    def _apply_loras(self, requested: tuple) -> None:
        from nunchaku.lora.runtime import load_lora, lora_model_info, set_loras

        model = self._lora_model()
        try:
            info = lora_model_info(model)
        except TypeError as e:
            raise RuntimeError(
                f"{e}: LoRAs for this model need a newer nunchaku-rocm (update it with git pull and reinstall)"
            ) from e
        loras = []
        for path, strength in requested:
            if path not in self._lora_cache:
                self._lora_cache[path] = load_lora(comfy.utils.load_torch_file(path, safe_load=True), info)
            loras.append((self._lora_cache[path], strength))
        set_loras(model, loras)
        self._applied_loras = requested
        # drop cached LoRAs no longer requested
        keep = {p for p, _ in requested}
        self._lora_cache = {p: w for p, w in self._lora_cache.items() if p in keep}
