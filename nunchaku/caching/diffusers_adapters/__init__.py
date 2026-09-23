"""
Pipeline adapters for first-block caching in Nunchaku models.
"""

from diffusers import DiffusionPipeline


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    """
    Apply caching to a diffusers Flux pipeline using ``NunchakuFluxTransformer2DModelV2``.

    Parameters
    ----------
    pipe : DiffusionPipeline
        The diffusers pipeline to apply caching to. Its ``.transformer`` must
        be a :class:`~nunchaku.models.transformers.transformer_flux_v2.NunchakuFluxTransformer2DModelV2`.
    *args
        Variable positional arguments passed to :func:`.flux_v2.apply_cache_on_pipe`.
    **kwargs
        Variable keyword arguments passed to :func:`.flux_v2.apply_cache_on_pipe`.
        Common arguments include:

            - ``residual_diff_threshold`` (float): Similarity threshold for cache validity
            - ``use_double_fb_cache`` (bool): Whether to use double first-block caching

    Returns
    -------
    DiffusionPipeline
        The same pipeline instance with caching applied.

    Raises
    ------
    AssertionError
        If the input is not a DiffusionPipeline instance.

    Notes
    -----
    This function modifies the pipeline in-place and returns the same instance.
    """
    assert isinstance(pipe, DiffusionPipeline)

    from .flux_v2 import apply_cache_on_pipe as apply_cache_on_pipe_fn

    return apply_cache_on_pipe_fn(pipe, *args, **kwargs)
