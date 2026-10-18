# Source this file before launching Stage-1 training.
# Exact environment from the validated fd7800d0633a snapshot.
export RAE_MATMUL_TF32=1
export RAE_BLAS_LIBRARY=cublas
export RAE_COMPILE_REGION=decoder_blocks
export RAE_COMPILE_MODE=max-autotune-no-cudagraphs
export RAE_COMPILE_DISC_BLOCKS=1
export RAE_AUX_COMPILE_MODE=max-autotune-no-cudagraphs
export RAE_LPIPS_LAYOUT=full
export RAE_DISABLE_DONATED_BUFFER=1
export RAE_COMPILE_LPIPS_COMPARE=1
export RAE_LPIPS_COMPILE_MODE=default
export RAE_COMPILE_DISC_NORM=1
export RAE_DISC_NORM_COMPILE_MODE=default
export RAE_LOCAL_ATTN_BACKEND=cudnn
export RAE_DISC_ATTN_BF16=1
export RAE_DECODER_ATTN_BF16=1
export RAE_DECODER_MLP_BF16=1
export RAE_DECODER_PROJ_BF16=1
export RAE_DISC_MLP_BF16=1
export RAE_DISC_PROJ_BF16=1
