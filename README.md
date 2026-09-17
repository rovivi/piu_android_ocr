# piu-ocr · Android

Lectura de pantallas de resultado de **Pump It Up** en el teléfono, sin LLM.
YOLO en NCNN + OCR clásico en C++ + matching contra el catálogo en Kotlin.
Un AAR de ~22 MB con las cuatro ABIs (`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`), sin dependencias en la app. Las cuatro y no solo arm64: el bundle de Play exige que todos los módulos compartan el set de ABIs, y la app que lo consume ya trae esas cuatro.

```kotlin
val ocr = PiuOcr.create(context)
val r = ocr.read(bitmap)          // r.song / r.level / r.chartType / r.score: Field(value, confidence, reason)
if (r.needsVlmCall) escalar(r)    // reason != null => "no confío, preguntá al VLM"
```

![antes / después](docs/img/before_after.png)

| campo | acierto end-to-end | precisión cuando responde |
|---|---|---|
| **score** | **0.905** | 0.97 |
| chart type | 0.767 | 0.79 |
| canción | 0.733 | 0.89 |
| nivel | 0.714 | 0.79 |

45 fotos de cabina con ground truth. Detalle, gráficas y qué falta:
**[docs/INFORME.md](docs/INFORME.md)**. Decisiones de diseño y constantes
calibradas: [MODULO_ANDROID.md](MODULO_ANDROID.md).

## Compilar

```bash
export JAVA_HOME=~/.jdks/ms-17.0.20.1 ANDROID_HOME=~/Android/Sdk    # JDK 17/21
gradle assembleRelease        # build/outputs/aar/piu-ocr-release.aar
```

## Verificar un cambio

```bash
tools/host/fetch.sh                                   # una vez: ncnn + opencv-mobile para Linux
python3 tools/parity/parity.py --detect --baseline    # C++ del .so vs Python vs ground truth
gradle testReleaseUnitTest                            # Kotlin == réplica Python, 90 filas
tools/parity/device.sh                                # lo mismo en un teléfono por adb
```

`tools/parity/README.md` explica las tres capas.

## Estructura

```
src/main/cpp/        detector (NCNN + TTA a mano), segment, text, badge, score, recognize, pipeline, jni_bridge
src/main/kotlin/     PiuOcr (gates, cruce con catálogo), SongMatcher (difflib porteado)
src/main/assets/     piu_yolo fp16, chars/level/digits .bin (int8), catalog.json
tools/host/          el mismo C++ como binario Linux
tools/parity/        comparador, baseline, gráficas
src/test/            ParityTest.kt (JVM)      src/androidTest/   DeviceParityTest.kt
```
