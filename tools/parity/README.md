# Test de paridad

Herramienta del test de regresión del módulo. Corre el **mismo C++ que va al
AAR** (compilado como binario Linux con `tools/host/`) sobre las fotos de
`piu_ocr/dataset_v2` y lo compara con `ResultReader` de Python y con el ground
truth.

La guía completa de verificación (las tres capas, el layout que espera, el
baseline y la validación sintética) está en
[`docs/VERIFICACION.md`](../../docs/VERIFICACION.md). Acá va lo mínimo:

```bash
tools/host/fetch.sh                                   # una vez: prebuilts Linux
python3 tools/parity/parity.py --detect --baseline    # falla si alguna métrica cae
./gradlew testReleaseUnitTest                         # Kotlin == réplica Python
tools/parity/device.sh                                # el .so arm64 real, por adb
```

Piezas:

- `threshold_sweep.py` — barrido de umbrales (gates) del `interpret()` sobre el
  JSON real del device + GT validado, con reporte de falsos positivos. No hace
  inferencia: funciones puras del JSON + catálogo.

| qué | dónde |
|---|---|
| binario de host (`piuocr_cli`) | `tools/host/` → `build_host/piuocr_cli` |
| comparador | `tools/parity/parity.py` (se relanza solo con el venv de `piu_yolo` si falta cv2) |
| matcher de referencia | `piu_ocr/song_match.py` si está al lado; si no `tools/parity/song_match_ref.py` |
| métricas de referencia | `tools/parity/baseline.json` |
| gráficas del informe | `tools/parity/report.py` → `docs/img/*.png` |
| animaciones del README | `tools/parity/animate.py` → `docs/img/*.gif` |
| fixture para Kotlin | `src/test/resources/parity_fixture.json` → `ParityTest.kt` |

Sin el dataset al lado, el fixture igual se puede resincronizar desde el `native`
ya guardado: `python3 tools/parity/parity.py --regen-expected`.

## Diferencias que quedan **a propósito** (nativo ≠ Python)

- `level`: nativo fuerza 2 dígitos en la bolita (`segmentBadge(..., 2)`), Python
  no. Nativo 0.872 de precisión vs 0.795.
- texto crudo: ~29 % de las fotos difieren en un carácter (empates en el
  producto punto con float32 en orden distinto). El match al catálogo lo absorbe:
  acuerdo de canción 0.978.
