Basic Usage
===========

The following is a minimal script for running 4-bit `FLUX.1 <github_flux_>`_ using Nunchaku.
Nunchaku provides the same API as `Diffusers <github_diffusers_>`_, so you can use it in a familiar way.

.. literalinclude:: ../../../examples/v1/flux.1-dev.py
   :language: python
   :caption: Running FLUX.1-dev (`examples/v1/flux.1-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-dev.py>`__)
   :linenos:

The key difference when using Nunchaku is replacing the standard ``FluxTransformer2dModel``
with :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`.
The :meth:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2.from_pretrained`
method loads quantized models and accepts either Hugging Face remote file paths or local file paths.

.. note::

   The :func:`~nunchaku.utils.get_precision` function automatically detects whether your GPU supports INT4 or FP4 quantization.
   Use FP4 models for Blackwell GPUs (RTX 50-series) and INT4 models for other architectures.
