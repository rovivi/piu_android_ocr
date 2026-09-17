plugins {
    id("com.android.library") version "8.7.3"
    kotlin("android") version "2.0.21"
}

android {
    namespace = "com.piu.ocr"
    compileSdk = 35
    // opencv-mobile linkea -static-openmp y el libomp del NDK 27 no trae
    // __kmpc_dispatch_deinit. El 29 sí.
    ndkVersion = "29.0.14206865"
    defaultConfig {
        minSdk = 24
        // Reglas que viajan al consumidor: R8 no debe renombrar PiuOcr ni sus
        // métodos native, el .so los busca por nombre (Java_com_piu_ocr_PiuOcr_*).
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Las CUATRO ABIs, y no por gusto: el bundle de Play exige que TODOS los
        // módulos soporten el mismo set, y la app base ya trae arm64-v8a,
        // armeabi-v7a, x86 y x86_64 de sus propias dependencias. Con solo arm64
        // acá, `bundleRelease` falla ("All modules with native libraries must
        // support the same set of ABIs") y no hay .aab que subir.
        //
        // Costo: el .so se triplica (~7.6 MB por ABI, ~30 MB el AAR). Si algún
        // día se quiere recortar, hay que bajar TAMBIÉN el set de la app base
        // (`composeApp`), nunca solo este: v7 + arm64 allá y acá deja el bundle
        // en arm64 + v7 sin x86 (emulador) ni x86_64.
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64") }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    buildTypes {
        release {
            // sin esto el .so viaja con símbolos de debug: 12.8 MB contra 2.5
            ndk { debugSymbolLevel = "none" }
        }
    }
    sourceSets["main"].kotlin.srcDir("src/main/kotlin")
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // Test de paridad (src/test): corre interpret() + SongMatcher en la JVM
    // sobre salidas grabadas del CLI de host. org.json real porque el de
    // android.jar es un stub.
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    // Test en device (src/androidTest): tools/parity/device.sh
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

// kotlinOptions está deprecado en Kotlin 2.x.
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
