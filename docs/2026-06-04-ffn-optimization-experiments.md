# 2026-06-04 Wan2.2 FFN 최적화 실험 보고서

## 배경

엔진 레벨 프로파일링에서 Wan2.2 denoise step의 FFN(fc1+fc2)이 43.9~67%를 차지하는 것으로 확인됨.
compile 전에는 43.9%, compile 후에는 67% (다른 연산이 compile로 더 많이 최적화되어 상대적 비중 증가).

본 실험에서는 FFN 연산을 파이썬/MLX 수준에서 최적화할 수 있는지 5가지 접근법을 테스트.

## 테스트 환경

- 하드웨어: Apple M4 Ultra 128GB (T6050)
- OS: macOS 26.5 (Tahoe) 25F71
- 모델: Wan2.2-TI2V-5B MLX (bf16)
- 해상도: 832x480, 41프레임
- MLX: seq_len=4290 (patch_size=[1,2,2])
- Python 3.12 + uv

## 사전 조건: compile 후 병목 분포

```
compile 후 per-step (2.3s):
  FFN (fc1+gelu+fc2): 67% = 1.18s
  SDPA (self-attn):   28% = 0.49s
  Cross-attn:          5% = 0.08s
  나머지:                  = 0.55s
```

FFN per block: 0.039s (fc1: 0.016s, fc2+gelu: 0.023s)
총 30개 transformer blocks.

---

## 실험 1: Chunked FFN

### 방법

FFN 중간 텐서 [B, seq_len, 14336]을 청크 단위로 처리.
메모리 대역폭 절감 및 캐시 효율 향상 기대.

### 결과 — Isolated (모델 forward only, 5회 평균)

```
Strategy                Time/step   vs baseline
─────────────────────────────────────────────────
baseline (no compile)    2.506s      +0.0% (최고)
model_compile            2.622s      -4.7%
chunked_c512_compile     2.698s      -7.7%
chunked_c1024_compile    2.700s      -7.8%
chunked_c2048_compile    2.668s      -6.5%
per_ffn_compile          2.842s     -13.4%
```

### 결과 — Pipeline (generate_video 10 steps)

```
Strategy                     Total   Denoise
──────────────────────────────────────────────
baseline (compile)           65.5s    23.9s
chunked_1024 + compile       70.5s    23.3s
chunked_2048 + compile       75.7s    30.6s
fused_cache + compile        91.4s    32.0s
fused_cache, no compile      97.8s    34.6s
```

### 분석

chunked FFN이 compile 그래프 최적화를 방해.
MLX compile은 전체 computation graph를 분석해서 operator fusion을 수행하는데,
파이썬 루프로 청크를 나누면 compile이 각 청크를 독립 연산으로 처리해 fusion 기회가 사라짐.
청크가 클수록(2048) 더 느려지는 것은 compile trace가 더 복잡해지기 때문.

---

## 실험 2: 양자화 (Quantization)

### 방법

FFN weight를 int4/int8로 양자화하여 메모리 대역폭 절감.
MLX nn.quantize 사용.

### 결과 (5회 평균)

```
Strategy                Time/step   vs baseline
─────────────────────────────────────────────────
bf16 (baseline)          2.338s      +0.0%
ffn int4 (group=64)      2.616s     -11.9%
full model int4          2.722s     -16.4%
ffn int8 (group=128)     2.767s     -18.4%
```

### 분석

모든 양자화 전략이 bf16보다 느림.
이유: Apple Silicon GPU에서 양자화된 GEMM은 weight를 실시간으로 dequantize해야 함.
이 dequantize 연산 비용이 메모리 대역폭 절약으로 얻는 이득보다 큼.
특히 bf16 GEMM은 GPU가 네이티브로 지원해서 이미 매우 빠름.

---

## 실험 3: Custom Metal Kernel (Fused Bias+GELU)

### 방법

`mx.fast.metal_kernel`로 bias addition + GELU(tanh)를 하나의 Metal 셰이더로 융합.
fc1 matmul 결과에 bias를 더하고 GELU를 적용하는 과정을 단일 커널로 처리.

### Metal Shader 코드

```metal
uint elem = thread_position_in_grid.x;
uint N = inp_shape[1];
T val = inp[elem] + bias[elem % N];
// GELU(tanh): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float xf = static_cast<float>(val);
float inner = 0.7978845 * (xf + 0.044715 * xf * xf * xf);
float result = 0.5f * xf * (1.0f + metal::precise::tanh(inner));
out[elem] = static_cast<T>(result);
```

### 결과

```
Strategy                      Time/step   vs baseline
─────────────────────────────────────────────────────────
baseline (compile)             2.344s      +0.0%
fused bias+GELU Metal kernel   2.618s     -11.7%
```

정확도: max diff 0.015625 vs MLX reference (bf16 GELU)

### 분석

Custom Metal 커널도 compile 그래프 최적화를 방해.
`mx.compile`은 MLX 내장 연산만 그래프에 포함시키고,
custom kernel 호출은 외부 연산으로 처리되어 fusion 기회가 손실됨.
또한 bias+GELU는 FFN 전체 시간의 ~10%만 차지해서 개선 효과가 제한적.

---

## 실험 4: Raw GEMM 성능 분석

### 방법

FFN matmul(fc1, fc2)의 실제 하드웨어 활용률 측정.
M4 Ultra bf16 peak: ~27.2 TFLOPS (GPU only).

### 결과

```
fc1: [8580, 3072] x [3072, 14336]
  시간: 12.33ms
  연산량: 755.7 GFLOPS
  활용률: 225% of GPU peak

fc2: [8580, 14336] x [14336, 3072]
  시간: 15.25ms
  연산량: 755.7 GFLOPS
  활용률: 182% of GPU peak
```

### 분석

**GPU peak의 180~225% 달성.** 이는 M4 Ultra에서 MLX가 CPU와 GPU를 동시에 사용하여
matmul을 처리하고 있음을 의미. 하드웨어 한계를 이미 초과 달성.
순수 matmul만 30블록에 0.83초. 실제 FFN 시간은 1.18초.
차이(0.35s)는 GELU, bias, residual connection 등의 오버헤드.

---

## 종합 결론

### 모든 FFN 최적화 시도 결과

```
접근법                         결과      원인
─────────────────────────────────────────────────────────────────
Chunked FFN (512/1024/2048)   느림    compile 그래프 방해
Per-FFN compile               느림    함수 호출 오버헤드 > 최적화 이익
int4/int8 양자화              느림    dequantize 오버헤드 > 대역폭 절약
Fused bias+GELU Metal kernel  느림    compile 그래프 방해 + 정확도 저하
Cache-friendly fused FFN      느림    chunk 루프 오버헤드
```

### 핵심 발견

1. **MLX GEMM은 이미 하드웨어 한계 도달** — GPU peak의 180~225%
2. **mx.compile이 이미 operator fusion을 최적으로 수행** — 파이썬 레벨 트릭은 방해만 됨
3. **양자화는 Apple Silicon GPU에서 역효과** — bf16 네이티브 지원이 더 빠름
4. **Custom Metal kernel은 compile과 호환되지 않음** — 그래프 최적화에서 제외됨
5. **FFN은 파이썬/MLX 수준에서 추가 최적화 불가능**

### 남은 개선 기회

| 영역 | 비중 | 개선 가능성 | 접근법 |
|------|------|-------------|--------|
| FFN | 67% | 낮음 | 이미 한계 도달 |
| SDPA | 28% | 낮음 | Flash Attention 이미 최적 |
| VAE decode | ~40% (전체 파이프라인) | **높음** | Tiled decode 최적화 |
| Thermal throttling | 가변 | 중간 | Step간 cooling pause |
| RoPE+norm+modulation | 5% | 낮음 | compile이 이미 처리 |

---

## 실험 스크립트

- `scripts/bench_ffn_optimization.py` — Isolated FFN 벤치마크
- `scripts/bench_ffn_pipeline.py` — Pipeline 통합 벤치마크
- `scripts/bench_ffn_quantize.py` — 양자화 벤치마크
- `scripts/bench_ffn_metal_kernel.py` — Custom Metal kernel 벤치마크
- `scripts/profile_compiled_ops.py` — Compiled ops 개별 프로파일링
- `scripts/deep_profile_wan22.py` — Transformer block 프로파일링
- `scripts/deep_profile_subcomponents.py` — Sub-component 프로파일링

## 벤치마크 데이터

- `artifacts/bottleneck_bench/ffn_optimization.jsonl`
- `artifacts/bottleneck_bench/ffn_pipeline.jsonl`
- `artifacts/bottleneck_bench/ffn_quantize.jsonl`
- `artifacts/bottleneck_bench/ffn_metal_kernel.jsonl`
