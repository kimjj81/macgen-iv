# 2026-06-04 VAE Decode 최적화 실험 (2차)

## 배경

1차 실험에서 VAE decode 병목이 CausalConv3d의 프레임별 루프(123회 conv2d)에서 발생함을 확인.
2차에서는 MLX native 3D conv, Custom Metal kernel, depth-chunked conv2d를 테스트.

## 테스트 환경

- M4 Ultra 128GB, macOS 26.5, Wan2.2-TI2V-5B MLX bf16
- 해상도: 832x480, 41프레임 (latent: [1,11,30,52,48])

---

## 1. MLX Native 3D conv_general

MLX의 `conv_general`이 5D 입력을 받아서 3D convolution을 직접 수행 가능한지 확인.

```
해상도                  Native 3D    Per-frame 2D    비율
──────────────────────────────────────────────────────────────
small  [1,21,60,104,1024]   488ms       231ms       0.47x (2.1x 느림)
medium [1,41,120,208,512]   966ms       240ms       0.25x (4.0x 느림)
large  [1,41,240,416,512]  3888ms       815ms       0.21x (4.8x 느림)
```

**결론: Native 3D conv는 2-5x 느림.** MLX의 3D conv Metal 커널이 2D에 비해 최적화가 안 되어 있음.
청크 단위 temporal 처리도 도움 안 됨 (chunk8, chunk16 모두 동일하게 느림).

## 2. Custom Metal 3D Conv Kernel

`mx.fast.metal_kernel`로 프레임 루프를 GPU 셰이더 내부로 이동.
각 GPU 스레드가 하나의 출력 픽셀을 계산 (3x3x3 커널 순회).

```
small [1,21,60,104,1024]:
  Original: 373ms
  Metal:    22,756ms  (61x 느림)
  정확도:    max diff = 0.0000 (완벽 일치)
```

**결론: 61x 느림.** Naive per-pixel 접근은 메모리 액세스 패턴이 캐시에 전혀 맞지 않음.
3D conv를 제대로 하려면 tiled GEMM, shared memory 활용이 필요한데 `mx.fast.metal_kernel`은
elementwise 연산에 적합하고 복잡한 conv/GEMM에는 부적합.

## 3. Depth-Chunked Conv2d (핵심 발견)

기존: 각 출력 프레임 t마다 kd(=3)번의 conv2d를 개별 호출
개선: temporal 프레임을 청크로 묶어서 배칭, depth 위치별로 한 번의 conv2d 호출

```python
# 기존: T_out × kd = 123회 conv2d
for t in range(T_out):
    for d in range(kd):
        conv2d(x[:, t+d], weight[:, d])

# 개선: ceil(T_out/chunk_size) × kd 회 conv2d (청크당 3회)
for chunk in temporal_chunks:
    for d in range(kd):
        conv2d(stack(chunk frames), weight[:, d])  # batched
```

### 해상도별 최적 chunk size 및 결과

```
해상도 (H×W)            chunk   Original   Optimized   Speedup
──────────────────────────────────────────────────────────────────
30×52   (ups0, C1024)     8       88ms       63ms      1.40x
60×104  (ups1, C1024)     4      341ms      199ms      1.72x
120×208 (ups2a, C1024→512) 3      827ms      731ms      1.13x
120×208 (ups2b, C512→512)  3      520ms      413ms      1.26x
240×416 (ups3a, C512→256)  2     1163ms     1132ms      1.03x
240×416 (ups3b, C256→256)  2      627ms      549ms      1.14x
```

모든 해상도에서 개별 conv3d 연산 기준으로 개선 확인.
chunk size 결정 로직: `H×W > 40K → cs=2, > 10K → cs=3, > 3K → cs=4, else → cs=8`

### 전체 VAE decode 적용 결과

```
Baseline (원본 CausalConv3d):          14,192ms
Depth-chunked (conv3d만 교체):         15,842ms  (-12% 느려짐)
```

**전체 파이프라인에선 12% 느림.** 이유:
- VAE decoder에 `mx.eval()` 배리어가 있어서 MLX lazy evaluation 최적화가 제한됨
- depth-chunked가 한 청크에 여러 프레임을 묶어서 eval 사이에 더 큰 그래프 생성
- `mx.eval()` 제거 시 메모리 부족으로 13% 느려짐 (1차 실험에서 확인)

---

## 종합 결론

### 테스트한 모든 VAE 최적화 방법

```
방법                                    개별 연산    전체 VAE
──────────────────────────────────────────────────────────────────
MLX Native 3D conv_general              0.2-0.5x    (미테스트)
Custom Metal direct 3D conv kernel       0.02x       (미테스트)
Depth-chunked conv2d (cs=2-8)           1.03-1.72x  0.90x (느림)
Batched conv2d (전체 프레임)            OOM         OOM
mx.eval() 제거                          n/a         0.87x (느림)
bf16 → fp16                             1.00x       n/a
적응형 conv3d (해상도별 전략)           n/a         0.95x (느림)
```

### 병목의 본질

개별 conv3d 연산은 depth-chunked로 3-72% 개선 가능.
하지만 VAE decoder 전체에서는:
1. `mx.eval()` 배리어가 MLX 그래프 최적화를 제한
2. `mx.eval()` 없이는 메모리 압박으로 느려짐
3. 이 trade-off를 파이썬 수준에서 해결 불가

### 전체 파이프라인 적용 결과

```
Full pipeline (832x480, 41 frames, 10 steps, interval_cfg=5):

Configuration                    Total    Denoise   VAE
──────────────────────────────────────────────────────────────
Baseline (no compile)            68.5s    26.1s    33.0s (tiled)
Compiled model                   70.0s    26.7s    22.5s (tiled)

VAE standalone (random latents):
  Original:                      17.0s
  Depth-chunked:                 19.2s   (0.89x, 12% 느림)

VAE tiled (same as pipeline):
  Original:                      23.5s
  Depth-chunked:                 25.5s   (0.92x, 8% 느림)

Pipeline estimate:               0.96x (4% 느림)
```

**전체 파이프라인에서도 4-8% 느림.** 
개별 conv3d 연산은 빠르지만, VAE decode 전체에서는 mx.eval() 배리어와 
chunking 오버헤드가 상쇄. 특히 ups3 (240×416)에서 개선이 1.03x에 불과.

### 남은 옵션

1. **MLX upstream에 3D conv 최적화 기여** — 근본 해결
2. **mx.compile을 VAE에 적용** — eval 배리어 없이 그래프 최적화
3. **전체 파이프라인에서 depth-chunked 효과 측정** — denoise+VAE 합산

## 실험 스크립트

- `artifacts/bottleneck_bench/bench_native_3dconv.py` — MLX native 3D conv 벤치마크
- `scripts/bench_vae_metal_3dconv.py` — Custom Metal 3D conv 커널
- 1차 실험 스크립트는 `docs/2026-06-04-vae-optimization-experiments.md` 참조

## 벤치마크 데이터

- `artifacts/bottleneck_bench/vae_native_3dconv.jsonl`
- `artifacts/bottleneck_bench/vae_metal_3dconv.jsonl`
