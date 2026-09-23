ControlNets
===========

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/nunchaku/assets/control.jpg
   :alt: ControlNet Integration with Nunchaku

Nunchaku supports `FLUX.1-tools <blog_flux1-tools_>`_ from Black-Forest-Labs.

FLUX.1-tools
------------

FLUX.1-tools Base Models
^^^^^^^^^^^^^^^^^^^^^^^^

Nunchaku provides quantized FLUX.1-tools base models.
The implementation follows the same pattern as described in :doc:`Basic Usage <./basic_usage>`,
utilizing an API interface compatible with `Diffusers <github_diffusers_>`_
where the ``FluxTransformer2dModel`` is replaced with :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`.
The primary modification involves switching to the appropriate ControlNet pipeline.
Refer to the following examples for detailed implementation guidance.

.. tabs::

   .. tab:: FLUX.1-Canny-Dev

      .. literalinclude:: ../../../examples/v1/flux.1-canny-dev.py
         :language: python
         :caption: Running FLUX.1-Canny-Dev (`examples/v1/flux.1-canny-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-canny-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Depth-Dev

      .. literalinclude:: ../../../examples/v1/flux.1-depth-dev.py
         :language: python
         :caption: Running FLUX.1-Depth-Dev (`examples/v1/flux.1-depth-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-depth-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Fill-Dev

      .. literalinclude:: ../../../examples/v1/flux.1-fill-dev.py
         :language: python
         :caption: Running FLUX.1-Fill-Dev (`examples/v1/flux.1-fill-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-fill-dev.py>`__)
         :linenos:

   .. tab:: FLUX.1-Redux-Dev

      .. literalinclude:: ../../../examples/v1/flux.1-redux-dev.py
         :language: python
         :caption: Running FLUX.1-Redux-Dev (`examples/v1/flux.1-redux-dev.py <https://github.com/nunchux-ai/nunchaku/blob/main/examples/v1/flux.1-redux-dev.py>`__)
         :linenos:

.. note::

   FLUX.1-tools LoRAs and community ControlNets (e.g. ControlNet-Union-Pro) are
   not currently supported by :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`
   in this fork.
