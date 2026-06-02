# LTX2.3 SDPA 병목 최적화 조사 보고서

## 1. MLX SDPA 구현 분석 (현재 상태)

### 1.1 MLX v0.31.1의 SDPA 아키텍처

**`mx.fast.scaled_dot_product_attention`**은 이미 Flash Attention 스타일의 tiled IO-aware 구현을 사용합니다.

Metal metallib 분석 결과, 두 가지 SDPA 커널 패밀리가 존재합니다:

#### (A) `sdpa_vector_2pass` (기존 커널)
- 2-pass tiled attention: 첫 번째 패스에서 QK^T scores의 max/sum을 계산, 두 번째 패스에서 softmax 후 V 가중합
- `air.simd_sum.f32`, `air.fast_exp.f32`, `air.fmax.f32` 사용 → SIMD reduction 활용
- 지원 dtype: bfloat16, float16, float32
- 지원 head_dim: **64, 96, 128, 256** (템플릿 특화)
- 파라미터: `query_transposed`, `has_mask`, `do_causal`, `bool_mask`, `float_mask`, `has_sinks`, `blocks`
- **LTX2.3 (head_dim=128, bfloat16)은 이 커널을 사용**

#### (B) `steel_attention` (신형 NAX 커널, M5+)
- Metal Performance Primitives(MPP)의 `matmul2d_cooperative` 사용
- Cooperative tensor 기반 → GPU threadgroup 간 협력 연산
- `matmul2d_descriptor`: 16x32x16 타일 레이아웃
- `simdgroup.barrier`, `simd_shuffle_xor.f32` 활용
- bq64(beam_query=64), bk32/bk64(beam_key), bd64/bd128(beam_dim) 변형
- **M5 이상 칩에서만 활성화**

### 1.2 MLX SDPA의 제약사항

| 항목 | 상태 |
|------|------|
| Softmax 정밀도 | **float32 강제** (bfloat16 입력도 float32로 softmax) |
| Head dim 지원 | 64, 96, 128, 256만 네이티브 (나머지는 unfused fallback) |
| 최대 seq_len | ~65K 이상에서 GPU watchdog 타임아웃 (단일 dispatch >5초) |
| GQA 지원 | 지원 (k/v를 pre-tile하지 않아도 됨) |
| Mask | causal 문자열, boolean/additive array 지원 |

### 1.3 mlx_video LTX2.3의 SDPA 호출 경로

```
attention.py::scaled_dot_product_attention()
  → reshape: (B, seq, heads*dim_head) → (B, seq, heads, dim_head)
  → swapaxes: (B, seq, heads, dim_head) → (B, heads, seq, dim_head)
  → mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
  → swapaxes + reshape: (B, heads, seq, dim_head) → (B, seq, heads*dim_head)
```

**문제점:**
- RoPE가 SDPA 외부에서 Python 레벨로 처리됨 → `mx.fast.rope` 사용 안 함
- LTX2.3의 RoPE는 interleaved 방식으로 Python에서 구현 → `mx.fast.rope(traditional=True)`로 대체 가능

---

## 2. 최적화 방안 (난이도/효과순)

### 2.1 [알고리즘] mx.fast.rope 활용 (간단, 즉시 적용 가능)

**현재:** `mlx_video/models/ltx_2/rope.py`에서 Python 레벨로 RoPE 구현
**개선:** `mx.fast.rope(a, dims, traditional=True, ...)` 사용

`mx.fast.rope`는 Metal kernel로 컴파일된 최적화된 RoPE입니다.
LTX2.3은 interleaved RoPE를 사용하며, `mx.fast.rope`의 `traditional=True`가 이에 해당합니다.

하지만 LTX2.3의 RoPE는 3D position(t, y, x)을 사용하는 특수한 구조이므로,
`mx.fast.rope`가 직접 호환되는지 확인이 필요합니다.
LTX2.3의 `generate_freqs`가 생성하는 freqs_cis를 `freqs` 파라미터로 전달하면
호환 가능할 가능성이 있습니다.

**예상 효과:** SDPA 외부 RoPE 연산 시간 단축 (SDPA 자체는 무관)

### 2.2 [알고리즘] Block-Diagonal / Sparse Attention (가장 효과적)

512x512에서 seq_len=9216일 때 O(n²) 폭증이 핵심 문제입니다.

#### 2.2.1 VSA (Video Sparse Attention) - NeurIPS 2025
- **Wan2.1에 적용하여 attention 6x, end-to-end 31s→18s 속도 향상 보고**
- 계층적 2단계: coarse stage에서 tile pooling → critical token 식별 → fine stage에서 block attention
- 훈련+추론 모두 sparse attention 사용 → train-test mismatch 없음
- LTX2.3에 retrofitting 가능 (후속 파인튜닝 필요)
- **한계:** 훈련이 필요함. inference-only로는 품질 저하 가능

#### 2.2.2 Block-Diagonal Attention (현재 테스트됨)
- 이미 5.1x 빠르다는 결과 확인됨
- 구현: spatial token을 블록 단위로 분할, 각 블록 내에서만 attention
- **권장:** 
  - 품질 평가 먼저 수행 (FID, FVD 메트릭)
  - 임계 해상도(예: 384x384 이하)에서만 사용
  - 블록 크기를 가변적으로 조정 (작은 해상도는 전체, 큰 해상도는 block-diagonal)

#### 2.2.3 Local Window + Global Token Attention
- 대부분의 attention mass가 로컬 spatial 영역에 집중됨
- 윈도우 크기 32~64 tokens로 제한 + 1~2개 global token 유지
- MLX SDPA는 mask array를 지원하므로 window mask 생성 후 전달 가능
- **예상 효과:** seq_len=9216에서 3~8x 속도 향상

### 2.3 [MLX] mx.fast.rope + SDPA 통합 최적화

현재 LTX2.3의 attention 파이프라인:
```
Q = to_q(x) → q_norm(Q) → apply_rotary_emb(Q, pe) [Python RoPE]
K = to_k(ctx) → k_norm(K) → apply_rotary_emb(K, pe) [Python RoPE]
V = to_v(ctx)
out = sdpa(Q, K, V)
```

개선 방안:
```python
# 1. mx.fast.rope 사용
q = mx.fast.rope(q, dims=dim_head, traditional=True, 
                  base=theta, scale=scale, offset=offset)

# 2. 또는 custom Metal kernel로 QK-norm + RoPE + SDPA 통합
#    (MLX의 mx.fast.metal_kernel 활용)
```

### 2.4 [Metal] mlx-mfa (Metal Flash Attention) 라이브러리 사용

**`mlx-mfa`** (PyPI: `mlx-mfa==2.50.1`) - Apple Silicon 전용 Metal Flash Attention

주요 특징:
- V34 NAX 커널 (M5+): SeedVR2-small에서 **0.89x SDPA** (SDPA보다 빠름)
- D=128 causal long-N에서 **1.75x SDPA** 속도 향상 (M1 Max)
- Sliding window에서 **21x SDPA**
- Sparse attention 지원 (v2.33.1+)
- `attn_bias` 네이티브 지원

LTX2.3에 적용:
```python
import mlx_mfa  # auto-hook: mx.fast.scaled_dot_product_attention을 오버라이드
# 또는
from mlx_mfa import flash_attention
out = flash_attention(q, k, v, scale=scale)
```

**주의:** mlx-mfa는 주로 LLM(causal attention)에 최적화됨.
LTX2.3의 bidirectional self-attention 지원 여부 확인 필요.

### 2.5 [Metal] Custom Metal Kernel 작성

MLX의 `mx.fast.metal_kernel` API를 사용한 커스텀 attention kernel:

```python
# 개념적 예시
sdpa_kernel = mx.fast.metal_kernel(
    name="ltx_sdpa_optimized",
    input_names=["q", "k", "v", "pe_cos", "pe_sin"],
    output_names=["out"],
    source="""
    // tile 기반 attention with RoPE fusion
    // head_dim=128 특화, bidirectional
    // SIMD group matmul 활용
    """
)
```

**필요 최적화 기법:**
1. **Tile 크기 최적화:** Apple Silicon GPU의 32KB shared memory에 맞춘 tile 크기
2. **Register blocking:** Q tile을 register에 유지, K/V tile을 반복 순회
3. **Softmax 온라인 알고리즘:** max/sum을 두 패스로 나누지 않고 single-pass에 처리
4. **RoPE fusion:** QK-norm + RoPE를 SDPA kernel 내부에 융합

**권장 타일 크기:**
- BM (query tile) = 64, BN (key tile) = 64, BK (dim tile) = 32
- Apple Silicon의 SIMD group = 32 threads

### 2.6 [알고리즘] Linear Attention 대체 (연구 단계)

#### Attention Surgery (2025)
- Wan2.1 1.3B에 적용하여 attention FLOP 40% 감소, 품질 유지
- Hybrid attention: 일부 token은 softmax, 일부는 linear attention
- Post-training 방식 (재훈련 필요하지만 적은 GPU-day)

#### LinVideo (2025)
- O(n) linear attention으로 video diffusion 대체
- Data-free post-training framework

**적용 가능성:** LTX2.3에 post-training 적용 가능하나, MLX에 linear attention kernel 구현 필요

---

## 3. 해상도별 권장 전략

### 256x256 (seq_len=2304, SDPA=86ms/block)
- 현재 MLX SDPA로 충분 (이미 최적화됨)
- mx.fast.rope만 교체하여 추가 이득
- **목표:** 86ms → ~70ms (RoPE 최적화 포함 전체 attention 시간)

### 512x512 (seq_len=9216, SDPA=1377ms/block)
- **핵심 전략:** attention 연산량 자체를 줄여야 함
- 옵션 A: Block-diagonal attention (5.1x 빠름, 품질 평가 필요)
- 옵션 B: Local window attention (mask 기반, MLX SDPA 활용)
- 옵션 C: VSA-style sparse attention (훈련 필요)
- **목표:** 1377ms → ~200~300ms/block

### 1024x1024+ (seq_len=36864+)
- Sparse attention 필수
- 분할 처리 (tile-by-tile) + GPU watchdog 방지
- Stream 기안 multi-command 처리

---

## 4. MLX vs PyTorch/CUDA 성능 비교

| 측면 | CUDA (PyTorch) | MLX (Apple Silicon) |
|------|----------------|---------------------|
| Flash Attention | FlashAttention-2/3, 최적화됨 | sdpa_vector_2pass (유사하지만 덜 성숙) |
| Head dim 128 | 높은 성능 | 지원되나 메모리 대역폭 제약 |
| bf16 attention | 혼합 정밀도 최적화 | softmax는 float32 강제 |
| Seq > 8192 | 효율적 (HBM 대역폭) | GPU watchdog 위험, 메모리 압박 |
| Custom kernel | Triton, CUDA C | Metal compute (mx.fast.metal_kernel) |
| Cooperative 연산 | Tensor Core | SIMD group matmul (MPP) |

**핵심 차이:** Apple Silicon은 unified memory 구조로 메모리 대역폭이 CUDA GPU보다 낮음.
→ O(n²) attention에서 대역폭이 병목 → sparse/local attention이 더 중요

---

## 5. 실행 우선순위

| 우선순위 | 작업 | 난이도 | 예상 효과 | 품질 영향 |
|----------|------|--------|-----------|-----------|
| 1 | Block-diagonal attention 품질 평가 | 낮음 | 5.1x (512x512) | 평가 필요 |
| 2 | Local window attention (mask 기반) | 중간 | 3~8x | 낮음 |
| 3 | mx.fast.rope로 RoPE 교체 | 낮음 | 전체 ~5~10% | 없음 |
| 4 | mlx-mfa 통합 테스트 | 중간 | 1.2~1.8x (SDPA만) | 없음 |
| 5 | Custom Metal kernel (RoPE+SDPA fusion) | 높음 | 1.5~2x | 없음 |
| 6 | VSA sparse attention 적용 | 높음 | 6x (훈련 필요) | 낮음 |
| 7 | Linear attention (post-training) | 매우 높음 | 40% FLOP 감소 | 평가 필요 |

---

## 6. 핵심 결론

1. **MLX의 SDPA는 이미 Flash Attention 스타일의 tiled 구현**을 사용합니다.
   - `sdpa_vector_2pass` 커널로 IO-aware 연산 수행
   - head_dim=128, bfloat16을 네이티브 지원
   - CUDA의 FlashAttention-2와 알고리즘적으로 동등

2. **512x512에서의 1377ms는 O(n²)의 근본적 한계**입니다.
   - MLX SDPA 자체의 최적화로는 한계 (이미 잘 최적화됨)
   - attention 연산량 자체를 줄이는 알고리즘적 접근이 필수

3. **가장 실용적인 단기 해결책:**
   - Block-diagonal attention 품질 평가 → OK 시 즉시 적용
   - Local window mask를 MLX SDPA에 전달 → 품질 저하 최소화

4. **가장 유망한 중기 해결책:**
   - VSA (Video Sparse Attention) 방식의 trainable sparse attention
   - Wan2.1에서 6x 속도 향상 입증, LTX2.3에도 적용 가능

5. **Metal 레벨 추가 최적화:**
   - mlx-mfa 라이브러리로 SDPA 교체 (1.2~1.8x 가능)
   - custom Metal kernel로 RoPE+QK-norm+SDPA fusion (kernel launch 오버헤드 제거)
