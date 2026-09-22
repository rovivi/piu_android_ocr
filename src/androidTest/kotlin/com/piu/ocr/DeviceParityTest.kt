package com.piu.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
 * MIDE EL CAMINO DE LA APP, no solo el del módulo: decodifica con el mismo
 * tope (4096) que `OcrScoreReader` y hace UNA lectura por foto. Los cuartos de
 * vuelta de una foto girada son del módulo (`Engine::read`), y cuando los dio
 * las cajas ya vuelven en el marco de la foto.
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
                // UNA lectura, como la app: los cuartos de vuelta de una foto
                // girada los hace `Engine::read` por dentro (fallback de
                // rotación) y las cajas vuelven ya en el marco de la foto.
                val t1 = System.nanoTime()
                val raw = it.readRaw(bmp)
                val ms = (System.nanoTime() - t1) / 1e6
                val r = PiuOcr.interpret(raw, it.matcher)
                out.put(f.nameWithoutExtension, JSONObject().apply {
                    put("native", JSONObject(raw))
                    put("ms", ms)
                    put("song", r.song.value ?: JSONObject.NULL)
                    put("level", r.level.value ?: JSONObject.NULL)
                    put("chart_type", r.chartType.value ?: JSONObject.NULL)
                    put("score", r.score.value ?: JSONObject.NULL)
                    put("raw", r.rawTitle)
                    put("boxes", boxesOf(r))
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

    /** Qué cajas dio la lectura. */
    private fun boxesOf(r: Reading): JSONObject = JSONObject().apply {
        put("titles", r.titleBoxes.size)
        put("score", r.scoreBox != null)
        put("badge", r.badgeBox != null)
        put("screen", r.screenBox != null)
    }
}
