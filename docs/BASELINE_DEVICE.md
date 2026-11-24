# Baseline de device — 63 fotos nuevas (punto de partida)

Punto de partida (detector viejo) y resultado con el detector nuevo
(`yolo26_v4` de `piu_yolo`, 150 epochs) + gates ajustados, sobre las 63 fotos
de `piu_yolo/images/new` contra el ground truth validado de
`tools/validator/dataset_v2.txt`.

## Setup

- Device: `sdk_gphone16k_arm64` (emulador), Android 17, ABI `arm64-v8a`.
- Detector embebido (baseline): `piu_yolo.bin`
  (`sha256 73183acb951868f3b403a6ccdc2f388a50d1fddcb87c39d11a8e09c321628bae`).
- Detector nuevo: `yolo26_v4/weights/best.pt` → NCNN fp16 @1280
  (`sha256 9e9eb90b9e22b97346b2e305c43dc9cd…`, 19.4 MB vs 6.4 MB del viejo).
- Test: `com.piu.ocr.test` / `DeviceParityTest`, 63 fotos pusheadas por adb.
- Raw: `docs/device_results_new63_baseline.json` (viejo) y
  `docs/device_results_new63_v4_tuned.json` (nuevo + gates).

```bash
./gradlew -q assembleDebugAndroidTest
adb install -r -t build/outputs/apk/androidTest/debug/piu-ocr-debug-androidTest.apk
adb shell am instrument -w -e class com.piu.ocr.DeviceParityTest \
    com.piu.ocr.test/androidx.test.runner.AndroidJUnitRunner
```

`tools/parity/device.sh` no corre tal cual acá (espera JDK en `~/.jdks`,
`dataset_v2/` y `DATASET/` como hermanos inexistentes, y el `adb pull` del
`results.json` falla por permisos). Workarounds usados:

- `JAVA_HOME=~/Library/Java/JavaVirtualMachines/ms-17.0.20.1/Contents/Home`,
  `ANDROID_HOME=~/Library/Android/sdk`.
- Fotos desde `piu_yolo/images/new` (`.jpeg` → copia `.jpg`), pusheadas a
  `/sdcard/Android/data/com.piu.ocr.test/files/piu_parity/` (+ `chmod 777` del dir).
- `results.json` pre-creado por shell con `chmod 666` para poder bajarlo.

## Resultado (acierto = correcto / total, n=63)

| campo | viejo (base) | v4 gates default | **v4 gates afinados** |
|---|---|---|---|
| song | 0.524 (cob 0.571, prec 0.917) | 0.556 (cob 0.619, prec 0.897) | **0.571** (cob 0.635, prec 0.900) |
| chart_type | 0.905 (prec 0.905) | 0.968 (prec 0.968) | **0.968** (prec 0.968) |
| level | 0.556 (cob 0.825, prec 0.673) | 0.810 (cob 0.937, prec 0.864) | **0.810** (cob 0.937, prec 0.864) |
| score | 0.714 (cob 0.778, prec 0.918) | 0.714 (cob 0.778, prec 0.918) | **0.825** (cob 0.937, prec 0.881) |
| **total** | **170/252** | **192/252** | **200/252 (+17.6 %)** |

Latencia: 303 ms → 863 ms media (modelo `s` @1280 + TTA; emulador, no
representativo de teléfono). APK de test: 59 MB (antes 50 MB).

## Gates ajustados (2026-09-19)

- `MIN_SONG_MARGIN` 0.010 → **0.001**: recupera "Super Capriccio" (margen
  0.0017) sin agregar errores.
- `MIN_SCORE_MARGIN` 0.010 → **0.003**: recupera 7 scores correctos que se
  descartaban, a cambio de 3 malos (menos con 0.003 que con 0.0).
- Validado también sobre el fixture de 90 fotos (otro set): 282 → 285
  aciertos, song precision se mantiene 1.000. Réplica Python
  (`tools/parity/parity.py`) actualizada y fixture regenerado (`--regen-expected`);
  `testReleaseUnitTest` verde.

## Barrido de umbrales (2026-09-21)

`tools/parity/threshold_sweep.py` re-corre la réplica de `interpret()` con
distintos umbrales sobre el JSON crudo del device (v4) contra el GT validado.
Conclusiones:

- **Umbral de confianza del detector: NO.** Los 4 FP de song tienen conf
  0.86–0.92, igual que los correctos (0.69–0.93), y "Super Capriccio" es OK con
  conf 0.07. Subir el umbral pierde más correctos de los FP que mata (0.9
  destruye todo). Queda en 0.
- **Margen de canción: no separa.** Los FP ganan con margen 0.04–0.14 (alto);
  subirlo pierde "Super Capriccio" (margen 0.0017) y no mata FP.
- **Largo del raw: sí.** `o-C` → "BOCA" y `sgtm` → "STORM" son raws de 2 y 4
  caracteres. Piso relativo `MIN_RAW_LEN=4` (`min(4, len(nombre))`): mata
  `o-C`→BOCA, deja intacto el fixture de 90 (el título "B3" con raw exacto
  sobrevive porque el piso es relativo), y device queda:

| campo | antes | **con MIN_RAW_LEN=4** |
|---|---|---|
| song | 36 ok / 4 FP (prec 0.900) | **36 ok / 3 FP (prec 0.923)** |
| level | 51 ok (cob 0.937) | **50 (cob 0.921)** |
| total | 200/252 | **199/252** |

El −1 de total es el level 21 de "NightTheater", correcto solo porque el FP
"BOCA" comparte nivel por casualidad: colgado de una canción equivocada, se
escala con el resto.

- Los 3 FP de song que quedan son ambiguos honestos: `FULLsONG` matchea
  "8 6 - FULL SONG -" contra GT "Gargoyle - FULL SONG -" (título parcial),
  y "INFiNiTE ENERZY -Overdoze-" le gana a "INFINITE ENERGY -Overdose-" con
  margen 0.066 (gemelas del catálogo). No hay umbral que los separe; son
  candidatos para escalar al VLM con un criterio más fino.

## Observaciones

- **song**: cuando responde acierta 90 %, pero rechaza el 36 % (cobertura
  0.635). Es la mayor ganancia potencial (labels del detector).
- **level**: los fallos restantes son lecturas en la lane equivocada
  (21↔23), no del gate.
- **chart_type**: los 2 fallos son votos `halfdouble`/`double` contra GT `s`.
- Los números de `docs/INFORME.md` (song 0.733, score 0.905) son sobre el
  dataset viejo de 58 fotos: **no** comparables con estos 63.

