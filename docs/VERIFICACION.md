# Verificación

Tres capas de test más una validación sintética. La idea es una sola: **que lo
que corre el teléfono sea lo mismo que la referencia Python, y que ninguna
métrica caiga sin que alguien se entere.**

| capa | qué prueba | dónde | necesita |
|---|---|---|---|
| 1. host | C++ del `.so` == Python == ground truth | `tools/parity/parity.py` | dataset + venv + prebuilts Linux |
| 2. JVM | `PiuOcr.interpret()` == réplica Python | `gradle testReleaseUnitTest` | fixture (sin fotos) |
| 3. device | `.so` arm64 real + Kotlin real en el teléfono | `tools/parity/device.sh` | teléfono por adb |
| synth | el código nuevo bajo ángulo/luz conocidos | `tools/synth.py` + `--synth` | prebuilts Linux |

## Layout esperado

`tools/parity/parity.py` asume que este repo está en
`piu_ocr/android/`, con los datos y el pipeline de referencia como hermanos:

```
piu_ocr/
├── android/          ← este repo
├── dataset_v2/        boxes.json, gt_song.json, gt_level.json, gt_score.json, catalog.json
├── DATASET/           las fotos .jpg
├── build_mobile/      chars.npz, level.npz, digits.npz
└── piu_yolo/venv/     python con cv2 (el script se relanza solo con él)
```

Sin `dataset_v2` y las fotos, la capa 1 no corre. Lo que **sí** corre sin el
dataset es la capa 2 (JVM) y, tras `tools/host/fetch.sh`, la validación
sintética.

## Capa 1 — C++ de host contra Python

Corre el **mismo C++ que va al AAR** (compilado como binario Linux) sobre las
fotos de `dataset_v2` y lo compara con `ResultReader` de Python y con el ground
truth. Es el test de regresión de cualquier cambio en el módulo.

```bash
tools/host/fetch.sh                    # una vez: ncnn + opencv-mobile Linux
```

El CLI (`build_host/piuocr_cli`) se compila solo la primera vez que corre
`parity.py`.

```bash
# OCR con las cajas de PyTorch (aísla el port de la detección), lista las que difieren
python3 tools/parity/parity.py --diff

# + detector NCNN end-to-end, graba fixture y baseline
python3 tools/parity/parity.py --detect --fixture --update-baseline

# lo que corre antes de un commit: falla (exit 1) si alguna métrica cae
python3 tools/parity/parity.py --detect --baseline
```

### Qué reporta

Por campo (song / level / chart_type / score) y por camino:

- **cobertura** — respondió / total,
- **precisión** — correcto / respondido,
- **acierto** — correcto / total,

para Python y para el nativo con las mismas cajas, más el **acuerdo** nativo ==
Python (por campo y sobre el texto crudo), el **recall del detector** NCNN
contra las cajas de PyTorch + TTA y la **latencia** nativa en host.

### Banderas útiles

| bandera | efecto |
|---|---|
| `--detect` | además corre el YOLO de NCNN y mide end-to-end |
| `--diff` | lista las fotos donde nativo ≠ Python |
| `--json RUTA` | detalle por foto (lo consume `report.py`) |
| `--fixture` | graba `src/test/resources/parity_fixture.json` |
| `--baseline` | compara contra `tools/parity/baseline.json` y falla si algo cae |
| `--update-baseline` | regraba el baseline |
| `--tol` | tolerancia de la comparación con el baseline (default 0.0) |
| `--augs "1,0.83,0.83f"` | pasadas del TTA del detector |
| `--buckets` | acierto por campo dentro de cada condición de la foto |
| `--synth` | valida sobre pantallas sintéticas (ver abajo) |
| `--regen-expected` | reescribe `expected` del fixture desde el `native` guardado, sin fotos |
| `--from-device ARCHIVO` | compara un `results.json` bajado del teléfono |

Flags opt-in para A/B: `--rectify`, `--bin-mode`, `--badge-mode`,
`--title-variants`, `--title-boxes`. Se reenvían al CLI de host, así que miden
exactamente el mismo camino que `PiuOcr.setOptions`.

## Capa 2 — JVM (`ParityTest`)

```bash
./gradlew testReleaseUnitTest
```

`--fixture` graba, por foto, el JSON crudo que devolvería `nativeRead` más lo
que una réplica Python de `PiuOcr.interpret` saca de él. `ParityTest.kt` corre
el `interpret` **real** sobre ese mismo JSON y exige igualdad: si alguien toca
un gate en Kotlin y no en la réplica (o al revés), falla. Además mide acierto
contra el ground truth contra el baseline grabado, para que una "mejora" que
empeora no pase en silencio.

No necesita fotos ni el `.so`: corre sobre el fixture versionado. Si un gate
cambia, hay que **regenerar el fixture** para que la capa 2 vuelva a estar
sincronizada:

- **Con el dataset y el proyecto Python al lado:**
  `python3 tools/parity/parity.py --detect --fixture --update-baseline`.
- **Sin el dataset** (solo este repo): `python3 tools/parity/parity.py
  --regen-expected`. `expected` es función pura de `native` + catálogo, así que
  se recalcula desde el JSON crudo ya guardado, sin fotos ni `cv2`. Cuando el
  proyecto `piu_ocr` no está al lado, el matcher sale de
  `tools/parity/song_match_ref.py` (réplica local de `song_match.py`); con el
  paquete presente, `parity.py` usa el `Catalog` real.

`GateSweepTest` no es un test sino un volcado: para cada foto del fixture
escribe en `build/gate_sweep.tsv` el título real contra los candidatos del
matcher, para elegir un gate con datos en vez de a ojo.

## Capa 3 — Device

```bash
tools/parity/device.sh
```

Instala el test instrumentado (`piu-ocr-debug-androidTest.apk`), empuja las
fotos con ground truth a `/sdcard/.../piu_parity/`, corre `DeviceParityTest` (el
`PiuOcr` real: `.so` arm64 + Kotlin) y baja `results.json` a
`build/device_results.json`. Después `parity.py --from-device` reporta:

- acierto end-to-end contra ground truth, comparado con `baseline.native_e2e`
  (±1 foto);
- Kotlin en device vs la réplica Python sobre el mismo JSON (debe ser 0);
- `.so` arm64 vs CLI de host foto por foto (misma lógica, otra CPU: lo que
  difiera es fp16/orden de flotantes, no un bug — salvo que sea mucho);
- latencia real por foto.

Necesita `JAVA_HOME` (17/21) y `ANDROID_HOME` como para `assembleRelease`.

## Validación sintética (`--synth`)

Donde no hay fotos reales, `tools/synth.py` dibuja pantallas PIU Phoenix
sintéticas (título, bolita con el color del chart, score, rank) y las deforma de
forma conocida: perspectiva 0/10/20/30°, brillo 0.4–1.0, parche de glare y
ruido. Solo necesita PIL + numpy.

```bash
python3 tools/synth.py --build              # genera build/synth/ + gt.json
python3 tools/parity/parity.py --synth      # default vs baseline grabada
python3 tools/parity/parity.py --synth --rectify   # A/B con un flag
```

Chequea dos cosas: que con los flags puestos se lea el ground truth sintético
bajo cada condición, y que el **modo default no cambie** respecto de la primera
corrida (que representa "antes del cambio").

## Baseline de regresión

`tools/parity/baseline.json` guarda las métricas de referencia. `--baseline`
falla si alguna cae más de `--tol`. Para verlo cambiar con intención:

```bash
python3 tools/parity/parity.py --detect --baseline            # verificar
python3 tools/parity/parity.py --detect --baseline --update-baseline   # regrabar
```

Gráficas del informe (requiere `matplotlib` y un `docs/parity_detail.json`
fresco):

```bash
python3 tools/parity/parity.py --detect --json docs/parity_detail.json
python3 tools/parity/report.py
```

## Flujo recomendado antes de un commit

```bash
python3 tools/parity/parity.py --detect --baseline   # no puede caer ninguna métrica
./gradlew testReleaseUnitTest                        # Kotlin == réplica Python
```

Si tocaste un gate de Kotlin, regenerá el fixture en el mismo paso
(`--detect --fixture --update-baseline`) para que la capa 2 no quede stale. Si
tocaste el C++ del OCR, corré también la capa 3.
