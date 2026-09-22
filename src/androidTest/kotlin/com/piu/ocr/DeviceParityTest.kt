package com.piu.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Test de paridad EN DEVICE: corre el módulo real (AAR + .so arm64) sobre las
 * fotos que `tools/parity/device.sh` empuja al teléfono y graba, por foto, el
 * JSON crudo del .so, lo que interpretó Kotlin y la latencia. El script
 * después lo baja y `parity.py --from-device` lo compara contra el ground
 * truth, contra la réplica Python y contra baseline.json.
 *
 * Fotos:     <externalFilesDir>/piu_parity/ (los .jpg)
 * Resultado: <externalFilesDir>/piu_parity/results.json
 *
 * ESTE BUILD MIDE EL CAMINO DE LA APP, no solo el del módulo: decodifica con
 * el mismo tope (4096) que `OcrScoreReader`, y cuando la lectura vertical sale
 * vacía corre los dos cuartos de vuelta igual que la app. Cada corrida queda en
 * el JSON (ms de cada pasada y qué lectura habría elegido la app), para poder
 * comparar 3 pasadas TTA contra 2 sin volver a tocar el test.
 */
class DeviceParityTest {

    @Test
    fun readAllPhotos() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "piu_parity")
        val photos = dir.listFiles { f -> f.name.endsWith(".jpg", true) }?.sortedBy { it.name }
            ?: emptyList()
        assertTrue("sin fotos en $dir: correr tools/parity/device.sh", photos.isNotEmpty())

        val t0 = System.nanoTime()
        val ocr = PiuOcr.create(ctx)
        val loadMs = (System.nanoTime() - t0) / 1e6

        val out = JSONObject()
        ocr.use {
            for (f in photos) {
                val bmp = decodeLikeTheApp(f.path) ?: continue
                val t1 = System.nanoTime()
                val rawU = it.readRaw(bmp)
                val msU = (System.nanoTime() - t1) / 1e6
                val rU = PiuOcr.interpret(rawU, it.matcher)
                var msTotal = msU
                var which = "upright"
                var picked = rU
                var raw90: String? = null
                var raw270: String? = null
                if (rU.song.value == null && rU.level.value == null &&
                    rU.chartType.value == null && rU.score.value == null
                ) {
                    for (deg in intArrayOf(90, 270)) {
                        val turned = rotated(bmp, deg) ?: continue
                        try {
                            val t2 = System.nanoTime()
                            val raw = it.readRaw(turned)
                            val ms = (System.nanoTime() - t2) / 1e6
                            msTotal += ms
                            val r = PiuOcr.interpret(raw, it.matcher)
                            if (deg == 90) raw90 = raw else raw270 = raw
                            if (r.song.value != null || r.level.value != null ||
                                r.chartType.value != null || r.score.value != null
                            ) {
                                which = deg.toString()
                                picked = r
                                break
                            }
                        } finally {
                            turned.recycle()
                        }
                    }
                }
                out.put(f.nameWithoutExtension, JSONObject().apply {
                    put("native", JSONObject(rawU))
                    put("ms", msU)
                    put("ms_total", msTotal)
                    put("picked", which)
                    put("rot90", raw90?.let { JSONObject(it) } ?: JSONObject.NULL)
                    put("rot270", raw270?.let { JSONObject(it) } ?: JSONObject.NULL)
                    put("song", picked.song.value ?: JSONObject.NULL)
                    put("level", picked.level.value ?: JSONObject.NULL)
                    put("chart_type", picked.chartType.value ?: JSONObject.NULL)
                    put("score", picked.score.value ?: JSONObject.NULL)
                    put("raw", picked.rawTitle)
                    put("boxes", boxesOf(picked))
                    put("boxes_u", boxesOf(rU))
                    put("u_empty", rU.song.value == null && rU.level.value == null &&
                        rU.chartType.value == null && rU.score.value == null)
                    put("size", "${bmp.width}x${bmp.height}")
                })
                bmp.recycle()
            }
        }
        val res = JSONObject().put("load_ms", loadMs).put("device", android.os.Build.MODEL)
            .put("abi", android.os.Build.SUPPORTED_ABIS.joinToString(",")).put("rows", out)
        File(dir, "results.json").writeText(res.toString(1))
    }

    /** Los mismos 4096 de `OcrScoreReader`: lo medido es lo que corre en la app. */
    private fun decodeLikeTheApp(path: String): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        val longSide = maxOf(bounds.outWidth, bounds.outHeight)
        while (longSide / sample > 4096) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inSampleSize = sample
        }
        return BitmapFactory.decodeFile(path, opts)
    }

    /** El `rotated()` de `OcrScoreReader`. */
    private fun rotated(source: Bitmap, degrees: Int): Bitmap? = runCatching {
        Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            Matrix().apply { postRotate(degrees.toFloat()) },
            true,
        )
    }.getOrNull()

    /** Qué cajas dio la lectura: la señal que la app podría usar para no girar. */
    private fun boxesOf(r: Reading): JSONObject = JSONObject().apply {
        put("titles", r.titleBoxes.size)
        put("score", r.scoreBox != null)
        put("badge", r.badgeBox != null)
        put("screen", r.screenBox != null)
    }
}
