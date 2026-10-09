# Leer títulos en coreano

Un gabinete en modo coreano muestra el título en Hangul (패러독스, no Paradox).
Hoy el lector no lo lee: `chars.bin` tiene 81 clases, todas latinas. Lo demás
ya está listo para el día en que tenga plantillas Hangul.

## Ya hecho (2026-10-09)

| pieza | qué |
|---|---|
| `pipeline.cpp` `appendUtf8` / `esc` | una etiqueta de `chars.bin` es un código Unicode; el raw sale en UTF-8 y el JSON como `\uXXXX` del código. Antes `char(L)` cortaba todo lo > 255. Con plantillas ASCII el resultado es el mismo byte de siempre. |
| `SongMatcher.normalize` | NFD → sin marcas → **NFC** → conserva `가-힣`. En latino el NFC no cambia nada (ParityTest intacto). |
| `SongMatcher.match` | una lectura con Hangul se compara contra `"k"`, una latina contra `"n"` como siempre. El candidato sale con `"n"`, que es el nombre que la app conoce. |
| `catalog.json` `"k"` | título coreano de las 654 canciones activas (`tools/catalog_kr.py`). Las 21 sin `"k"` son K-pop quitado en PHOENIX 2. |
| `tools/parity/song_match_ref.py` | mismo `normalize` que Kotlin. |
| `SongMatcherKoreanTest` | Hangul exacto y con una sílaba mal leída; `normalize` latino sin cambios. |

La app (PumpItUpTracker) ya empata contra `songNameKr` también en
`OcrSongMatch.kt`, así que una lectura Hangul llega a la canción por los dos lados.

## Lo que falta: plantillas

1. **Fotos.** Pantallas de resultado de un gabinete en coreano. 30–50 para
   empezar, con el título en ground truth como en `dataset_v2`.
2. **Qué sílabas.** Los 638 títulos con Hangul usan **445 sílabas distintas**.
   Por frecuencia: 50 cubren el 61 % de las apariciones, 100 el 77 %, 200 el
   90 %, 300 el 96 %. No hace falta el bloque entero (11 172).
3. **Segmentación, el riesgo real.** `segmentChars` corta por huecos de
   columna. Una sílaba Hangul suele tener un hueco vertical entre la consonante
   inicial y la vocal (가 = ㄱ | ㅏ), así que puede partir una sílaba en dos
   glifos. Medir con las fotos antes de generar plantillas: si parte, unir
   cortes contiguos cuya caja conjunta sea ~cuadrada (las sílabas lo son;
   el latino es angosto) solo cuando la caja sea más alta que una minúscula.
4. **Plantillas.** Recortar los glifos con el mismo `GLYPH_W×GLYPH_H` (24×32),
   etiquetar con el código Unicode y agregarlos a `chars.bin` (la clase es un
   `int32`, ya cabe). Medir con `tools/parity/parity.py --detect --baseline`
   que el latino no cayó: más clases = más confusiones posibles.
5. **AAR nuevo** → `PumpItUpTracker/ocr/libs/piu-ocr-release.aar`.

Para regenerar `"k"` después de un catálogo nuevo:

```bash
# en ligas-piu-api (piu_songs.song_name_kr lo llena el sync de PIUGame)
select json_object_agg(song_name, song_name_kr) from piu_songs where song_name_kr is not null;
python3 tools/catalog_kr.py kr_titles.json
```
