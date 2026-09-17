# Documentación

Índice de la documentación del módulo. El punto de entrada sigue siendo el
[README raíz](../README.md); esto es el detalle.

## ¿Qué buscás?

| quiero… | leé |
|---|---|
| entender la API y usarla desde mi app | [API.md](API.md) |
| entender cómo está hecho y por qué | [ARQUITECTURA.md](ARQUITECTURA.md) |
| ver los números medidos y qué falta | [INFORME.md](INFORME.md) |
| correr los tests y verificar un cambio | [VERIFICACION.md](VERIFICACION.md) |
| el plan de mejora bajo luz/ángulo | [PLAN_LUZ_ANGULO.md](PLAN_LUZ_ANGULO.md) |

## Los documentos

- **[API.md](API.md)** — API pública de Kotlin (`PiuOcr`, `Reading`, `Field`,
  cajas), integración en Gradle, dynamic feature de Play y política de confianza
  (`needsVlmCall`). Sin detalles internos.

- **[ARQUITECTURA.md](ARQUITECTURA.md)** — el diseño por dentro: qué corre en
  C++ y qué en Kotlin, el contenido medido del AAR, el detector con TTA, el OCR
  clásico, las **constantes calibradas** y los **gates**, los flags opt-in, el
  build y las trampas conocidas. Es el documento que hay que leer antes de tocar
  código.

- **[INFORME.md](INFORME.md)** — informe de estado: antes/después, paridad con
  el pipeline Python, el efecto del gate, el score, el TTA y los bugs que
  encontró el test de paridad. Todos los números salen de
  `tools/parity/parity.py` sobre las 45 fotos con ground truth.

- **[VERIFICACION.md](VERIFICACION.md)** — how-to de las tres capas de test
  (C++ de host vs Python, `ParityTest` en la JVM, test instrumentado en device)
  más la validación sintética y el baseline de regresión.

- **[PLAN_LUZ_ANGULO.md](PLAN_LUZ_ANGULO.md)** — estrategia opt-in para leer
  mejor bajo luz irregular y ángulos (rectify, CLAHE/adaptive, badge adaptativo,
  variantes de título). Estado de cada fase y cómo medir el A/B.

## Recursos

- `img/` — gráficas y animaciones referenciadas por el [README raíz](../README.md)
  y por [INFORME.md](INFORME.md).
  - PNGs de informe: regenerados por `tools/parity/report.py`.
  - GIFs (`field_scan.gif`, `synth_conditions.gif`): generados por
    `python3 tools/parity/animate.py`. El hero usa la foto real
    `imagtest.jpg` de la raíz del repo; si no está, cae a la pantalla sintética.
- `parity_detail.json` — detalle por foto generado por
  `python3 tools/parity/parity.py --detect --json docs/parity_detail.json`.
  Es un artefacto: no editar a mano.
- El README de la herramienta de paridad sigue colocado con el código, en
  [`tools/parity/README.md`](../tools/parity/README.md).
