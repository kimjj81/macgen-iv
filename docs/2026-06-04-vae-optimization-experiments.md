# 2026-06-04 VAE Decode 프로파일링 및 최적화 실험

## 배경

Wan2.2 파이프라인에서 VAE decode가 전체 시간의 ~40%를 차지 (16초/40초).
FFN 최적화가 한계에 도달한 후, VAE decode 병목을 식별하고 개선 시도.

## 테스트 환경

- 동일 환경 (M4 Ultra 128GB, macOS 26.5, Wan2.2-TI2V-5B MLX bf16)
- 해상도: 832x480, 41프레임
- Latent: [1, 11, 30, 52, 48]

## 1. VAE 프로파일링 결과

### 레이어별 시간 분배

```
VAE Decode 총 시간: ~13.0초 (프로파일링 오버헤드 제외 시 ~16초)

conv2 (1x1x1):        1ms    (0.0%)
decoder.conv1:       17ms    (0.1%)
middle blocks:      334ms    (2.6%)  — ResBlock+Attn+ResBlock
upsample_0:         601ms    (4.6%)
upsample_1:       2,893ms   (22.3%)
upsample_2:       4,473ms   (34.5%)  ← 최대 병목
upsample_3:       4,561ms   (35.2%)
head:               234ms    (1.8%)
─────────────────────────────────────
                   13,115ms (100%)
```

### 상세 업샘플 블록 내부

```
Upsample 0 (30×52→60×104, temporal up):  601ms
  ResBlock×3:  156ms×3 = 468ms
  Resample:    109ms
  Shortcut:     18ms

Upsample 1 (60×104→120×208, temporal up):  2,893ms
  ResBlock×3:  726+683+669 = 2,078ms (72%)
  Resample:    622ms
  Shortcut:    158ms

Upsample 2 (120×208→240×416, spatial up):  4,473ms
  ResBlock_0:  1,633ms (37%) ← dim 변환 (1024→512)
  ResBlock_1:    825ms
  ResBlock_2:    797ms
  Resample:      814ms
  Shortcut:      353ms

Upsample 3 (240×416 유지, channel 축소):  4,561ms
  ResBlock_0:  2,417ms (53%) ← dim 변환 (512→256), 최대 병목
  ResBlock_1:  1,189ms
  ResBlock_2:  1,126ms
```

### 병목 원인 분석

각 ResidualBlock 내부: norm → silu → CausalConv3d → norm → silu → CausalConv3d.

CausalConv3d 구현이 **프레임별 루프**로 3D conv를 분해:
```python
for t in range(T_out):        # 41 iterations
    for d in range(kernel_d):  # 3 iterations
        conv2d(x[:, t+d], weight[:, d])  # 123 individual conv2d calls
```

**41프레임 × kernel_depth 3 = 123번의 개별 conv2d 호출.**
각 호출마다 MLX 커널 런치 + Python 오버헤드 + 동기화 비용 발생.

실제 conv2d 연산 시간은 2.8ms/회인데, 전체 ResBlock_0은 2,417ms.
순수 연산: 82 × 2.8ms = 230ms. 나머지 2,187ms (90%)는 오버헤드.

---

## 2. 최적화 실험

### 실험 A: Batched Conv2d (프레임 루프 제거)

모든 프레임을 하나의 배치로 묶어서 단일 conv2d 호출.

```
해상도              Original    Batched    Speedup
──────────────────────────────────────────────────
240×416 (ups3)      945ms      19,432ms   0.20x ← OOM
120×208 (ups2)      692ms         n/a
 60×104 (ups1)      330ms        100ms    3.30x
 30×52  (ups0)       77ms         n/a
```

결과: 큰 해상도에서는 메모리 부족으로 역효과.

### 실험 B: Chunked Batched Conv2d (청크 단위 배칭)

프레임을 4/8/16개 청크로 나누어 배칭.

```
해상도              Original   chunk4   chunk8   chunk16
─────────────────────────────────────────────────────────────
240×416 (ups3)       945ms    1,086ms  1,941ms  2,894ms
120×208 (ups2)       692ms      621ms    591ms    791ms  ← 1.17x
 60×104 (ups1)       336ms      195ms    131ms    142ms  ← 2.57x
 30×52  (ups0)        77ms       47ms     33ms     29ms  ← 2.68x
```

결과: 작은 해상도에서 최대 2.7x 빠름. 큰 해상도(ups3)에서는 느려짐.

### 실험 C: 적응형 Conv3d (해상도에 따라 자동 선택)

- H×W ≤ 25,000: chunked batched (chunk=8)
- H×W > 25,000: per-frame (원래 방식)

```
전체 VAE decode:
  Baseline:   15,896ms
  Adaptive:   17,443ms  (5.3% 느림)
```

결과: 작은 해상도의 개선이 큰 해상도의 오버헤드를 상쇄하지 못함.

### 실험 D: mx.eval() 제거

VAE decoder에 있는 중간 `mx.eval()` 호출 제거.

```
Baseline:          15,236ms
No eval (decoder): 17,262ms  (13% 느림)
```

결과: `mx.eval()`이 메모리 관리에 필수적. 제거하면 그래프가 너무 커져서 느려짐.

### 실험 E: bf16 vs fp16

```
모든 해상도에서 bf16 = fp16 (차이 없음)
```

Apple GPU에서 bf16과 fp16은 동일한 연산 유닛 사용.

---

## 3. 종합 결론

### VAE 병목의 본질

```
ups2 + ups3 = 70% of VAE time = ~9초
이 중 90%는 Python 레벨 프레임 루프 오버헤드:
  - 123회 conv2d 커널 런치
  - 각 호출의 Python→MLX bridge 오버헤드
  - 중간 텐서 할당/해제

순수 conv2d 연산: 10% (230ms out of 2,417ms)
```

### 실험 결과 요약

```
접근법                       결과        원인
────────────────────────────────────────────────────────────────
Batched conv2d (전체)        OOM        19GB 중간 텐서
Chunked conv2d (작은 해상도)  2.7x 빠름  GPU 활용률 향상
Chunked conv2d (큰 해상도)    느림      메모리 압박
적응형 Conv3d                5% 느림    오버헤드 상쇄 불가
mx.eval() 제거               13% 느림   메모리 관리 악화
bf16→fp16 전환               차이 없음   동일 하드웨어 유닛
```

### 남은 개선 옵션

1. **Custom Metal 3D Conv 커널** (유일한 실제 해결책)
   - 프레임 루프를 Metal 셰이더 내부로 이동
   - 예상: ups2+ups3에서 3-5x 속도 향상 → VAE 전체 50-60% 단축
   - MLX의 `mx.fast.metal_kernel`로 구현 가능
   - 하지만 3D conv Metal 구현은 복잡함 (공유 메모리 타일링, 경계 처리 등)

2. **MLX upstream에 3D conv 최적화 요청**
   - MLX의 `conv_general`이 3D conv를 네이티브로 지원하면 해결됨
   - 현재 CausalConv3d가 Python 루프로 분해하는 것이 근본 원인

3. **VAE weight 양자화** (denoise와 달리 VAE는 int4/int8이 도움될 가능성)
   - VAE는 메모리 대역폭 바운드 (큰 activation × 많은 conv)
   - weight를 4bit로 줄이면 대역폭 절반 → 가능성 있음

### 다음 단계 우선순위

1. VAE weight 양자화 테스트 (가장 간단, 가능성 높음)
2. Custom Metal 3D conv 커널 (가장 효과적, 구현 복잡)
3. MLX upstream 기여 (장기적)

## 실험 스크립트

- `scripts/profile_vae_decode.py` — VAE 레이어별 프로파일링
- `scripts/profile_vae_detail.py` — 업샘플 블록 상세 프로파일링
- `scripts/bench_vae_conv3d.py` — Batched conv3d 벤치마크
- `scripts/bench_vae_conv3d_v2.py` — Chunked conv3d 벤치마크
- `scripts/bench_vae_adaptive_v2.py` — 적응형 conv3d 전체 VAE 벤치마크
- `scripts/bench_vae_no_eval.py` — mx.eval 제거 테스트

## 벤치마크 데이터

- `artifacts/bottleneck_bench/vae_upsample_detail.jsonl`
- `artifacts/bottleneck_bench/vae_conv3d_opt.jsonl`
- `artifacts/bottleneck_bench/vae_conv3d_chunked.jsonl`
- `artifacts/bottleneck_bench/vae_adaptive.jsonl`
