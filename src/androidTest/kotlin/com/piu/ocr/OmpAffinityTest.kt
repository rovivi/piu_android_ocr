package com.piu.ocr

import android.graphics.Bitmap
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regresión del abort de libomp en el primer `cvtColor` de `nativeRead`
 * (`__kmp_create_affinity_none_places` -> KMP_ASSERT -> abort). Pasa en
 * cualquier device; para que el test signifique algo hay que reproducir la
 * condición: un emulador rooteado (google_apis, no playstore) con un core
 * apagado ANTES de lanzar el proceso:
 *
 *   adb root && adb shell 'echo 0 > /sys/devices/system/cpu/cpu3/online'
 *   ./gradlew connectedDebugAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=com.piu.ocr.OmpAffinityTest
 *
 * Sin el fix de JNI_OnLoad el proceso muere ("Process crashed") en vez de fallar.
 */
class OmpAffinityTest {

    @Test
    fun readSurvivesOfflineCpu() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val bmp = Bitmap.createBitmap(1280, 720, Bitmap.Config.ARGB_8888)
        PiuOcr.create(ctx).use { assertTrue(it.readRaw(bmp).startsWith("{")) }
        bmp.recycle()
    }
}
