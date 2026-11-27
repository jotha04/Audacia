/**
 * Hacu Inventario — Cliente de voz web y Push-to-Talk
 * Responsable: Integrante 4 (Hacu Inventario, API interna y voz web)
 *
 * Características principales:
 * - Push-to-Talk mediante barra espaciadora en PC (cuando no se está en un input/textarea)
 * - Botón táctil para móvil / tablet (tap o pulsar sostenido)
 * - Captura de micrófono con MediaRecorder API
 * - Transcripción con faster-whisper en el servidor
 * - Reproducción de audio TTS (Piper) EXCLUSIVAMENTE en el navegador del cliente
 * - Normalización fonética ("Audacia") y cero audio en altavoces del servidor
 * - Flujo de Pre-Confirmación obligatoria en dos pasos para mutaciones
 * - Expiración del contexto conversacional tras 2 horas de inactividad (7200 segundos)
 */

(function () {
    "use strict";

    // ─────────────────────────────────────────────────────────────────────────
    // CONFIGURACIÓN Y CONSTANTES
    // ─────────────────────────────────────────────────────────────────────────

    const CONFIG = {
        ENDPOINTS: {
            MENSAJE: "/api/haku/mensaje/",
            CONFIRMAR: "/api/haku/confirmar/",
            CANCELAR: "/api/haku/cancelar/",
            TRANSCRIBIR: "/api/haku/voz/transcribir/",
            SINTETIZAR: "/api/haku/voz/sintetizar/",
            SALUD: "/api/haku/salud/",
            ESTADO: "/api/haku/estado/",
            ALERTAS: "/api/haku/alertas/",
            PRODUCTO_CREAR: "/api/haku/producto/crear/",
            PRODUCTO_EDITAR: "/api/haku/producto/editar/",
            PRODUCTO_CONSULTAR_CODIGO_BARRAS: "/api/haku/producto/consultar-codigo-barras/",
            USUARIO_CREAR: "/api/haku/usuario/crear/",
            USUARIO_EDITAR: "/api/haku/usuario/editar/",
            PROPUESTA_ACTUALIZAR_DESTINATARIO: "/api/haku/propuesta/actualizar-destinatario/",
        },
        TIMEOUT_INACTIVIDAD_MS: 120 * 60 * 1000, // 2 horas (120 minutos / 7200 segundos)
        TIMEOUT_STT_MS: 45 * 1000, // Evita bloqueo indefinido del micrófono ante cortes LAN/móvil
        TIMEOUT_MENSAJE_MS: 60 * 1000,
        MIME_TYPES: [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/mp4;codecs=mp4a.40.2",
            "audio/mp4",
            "audio/ogg;codecs=opus",
            "audio/wav",
        ],
    };

    // ─────────────────────────────────────────────────────────────────────────
    // ESTADO DE LA APLICACIÓN
    // ─────────────────────────────────────────────────────────────────────────

    function generarUuidV4() {
        if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
            return crypto.randomUUID();
        }
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
        });
    }

    const state = {
        conversacionId: generarUuidV4(),
        grabando: false,
        iniciandoGrabacion: false,
        detencionPendiente: false,
        deteniendoGrabacion: false,
        timerPostRoll: null,
        procesando: false,
        reproduciendoAudio: false,
        audioElement: null,
        abortControllerTTS: null,
        interrupcionConsumida: false,
        mediaRecorder: null,
        audioChunks: [],
        streamMicrofono: null,
        audioContextMetro: null,
        analyserNode: null,
        maxRmsGrabacion: 0,
        animFrameMetro: null,
        tiempoInicioGrabacion: 0,
        propuestaActiva: null,
        ultimoAcceso: Date.now(),
        timerInactividad: null,
        espacioPresionado: false,
        ttsPlaybackId: 0,
        interactionActual: null,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // INSTRUMENTACIÓN TEMPORAL DE LATENCIA (CRONÓMETRO TÉCNICO END-TO-END)
    // ─────────────────────────────────────────────────────────────────────────

    function generarIdAleatorio(prefijo) {
        const rand = Math.random().toString(36).substring(2, 8);
        return `${prefijo}_${Date.now().toString(36)}_${rand}`;
    }

    function registrarTimingLog(interactionId, stage, durationMs, extra = {}) {
        let extraStr = "";
        for (const [k, v] of Object.entries(extra)) {
            if (v !== undefined && v !== null) {
                if (typeof v === "number") {
                    extraStr += ` ${k}=${v.toFixed(2)}`;
                } else {
                    extraStr += ` ${k}=${v}`;
                }
            }
        }
        console.log(`[HAKU-TIMING] interaction=${interactionId || "anon"} stage=${stage} duration_ms=${durationMs.toFixed(2)}${extraStr}`);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UTILIDADES CSRF Y RED
    // ─────────────────────────────────────────────────────────────────────────

    function getCsrfToken() {
        // 1. Buscar en meta tag
        const metaTag = document.querySelector('meta[name="csrf-token"]');
        if (metaTag && metaTag.content) return metaTag.content;

        // 2. Buscar en input oculto
        const inputCsrf = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (inputCsrf && inputCsrf.value) return inputCsrf.value;

        // 3. Buscar en cookies
        const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]*)/);
        return match ? decodeURIComponent(match[1]) : "";
    }

    function crearSenalTimeout(timeoutMs) {
        if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
            return { signal: AbortSignal.timeout(timeoutMs), limpiar: () => {} };
        }
        const controller = new AbortController();
        const timerId = setTimeout(() => controller.abort(), timeoutMs);
        return { signal: controller.signal, limpiar: () => clearTimeout(timerId) };
    }

    function esErrorTimeout(error) {
        return Boolean(
            error && (error.name === "TimeoutError" || error.name === "AbortError")
        );
    }

    async function apiFetch(url, options = {}) {
        const headers = options.headers || {};
        headers["X-CSRFToken"] = getCsrfToken();
        headers["X-Requested-With"] = "XMLHttpRequest";

        // Correlación de telemetría end-to-end
        if (!headers["X-Haku-Request-ID"]) {
            headers["X-Haku-Request-ID"] = generarIdAleatorio("req");
        }
        if (!headers["X-Haku-Interaction-ID"] && state.interactionActual && state.interactionActual.id) {
            headers["X-Haku-Interaction-ID"] = state.interactionActual.id;
        }

        const config = { ...options, headers };
        const response = await fetch(url, config);

        if (response.status === 401) {
            mostrarToast("Sesión expirada o no iniciada. Redirigiendo...", "error");
            setTimeout(() => { window.location.href = "/cuentas/login/"; }, 1500);
            throw new Error("No autenticado");
        }

        if (response.status === 403) {
            mostrarToast("Acceso denegado o cuenta desactivada.", "error");
            throw new Error("Acceso denegado");
        }

        return response;
    }


    // ─────────────────────────────────────────────────────────────────────────
    // GESTIÓN DEL TIMER DE 2 HORAS DE INACTIVIDAD (7200s)
    // ─────────────────────────────────────────────────────────────────────────

    function reiniciarTimerInactividad() {
        state.ultimoAcceso = Date.now();
        if (state.timerInactividad) clearTimeout(state.timerInactividad);

        state.timerInactividad = setTimeout(() => {
            manejarExpiracionInactividad();
        }, CONFIG.TIMEOUT_INACTIVIDAD_MS);
    }

    function manejarExpiracionInactividad() {
        state.propuestaActiva = null;
        ocultarTarjetaPropuesta();
        mostrarToast("El contexto de Hacu Inventario ha caducado por inactividad (2 horas). Tu sesión sigue abierta.", "info");
        agregarMensajeSistema("Contexto operativo reiniciado por inactividad (2 horas / 7200 s).");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MONITOREO DE NIVEL DE ENTRADA (AUDIO METER / ANALYSER)
    // ─────────────────────────────────────────────────────────────────────────

    function iniciarMedidorNivelAudio(stream) {
        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return;

            if (!state.audioContextMetro || state.audioContextMetro.state === "closed") {
                state.audioContextMetro = new AudioContextClass();
            }
            if (state.audioContextMetro.state === "suspended") {
                state.audioContextMetro.resume();
            }

            const source = state.audioContextMetro.createMediaStreamSource(stream);
            const analyser = state.audioContextMetro.createAnalyser();
            analyser.fftSize = 512;
            source.connect(analyser);

            state.analyserNode = analyser;
            state.maxRmsGrabacion = 0;

            const buffer = new Float32Array(analyser.fftSize);
            const medir = () => {
                if (!state.grabando) return;
                analyser.getFloatTimeDomainData(buffer);
                let sumSq = 0;
                for (let i = 0; i < buffer.length; i++) {
                    sumSq += buffer[i] * buffer[i];
                }
                const rms = Math.sqrt(sumSq / buffer.length);
                if (rms > state.maxRmsGrabacion) {
                    state.maxRmsGrabacion = rms;
                }
                state.animFrameMetro = requestAnimationFrame(medir);
            };
            state.animFrameMetro = requestAnimationFrame(medir);
        } catch (e) {
            console.warn("[HakuVoz] No se pudo inicializar AnalyserNode:", e);
        }
    }

    function detenerMedidorNivelAudio() {
        if (state.animFrameMetro) {
            cancelAnimationFrame(state.animFrameMetro);
            state.animFrameMetro = null;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GRABACIÓN DE VOZ (MICROFONO CLIENTE)
    // ─────────────────────────────────────────────────────────────────────────

    function esDispositivoTactil() {
        return Boolean(
            (window.matchMedia && window.matchMedia("(pointer: coarse)").matches)
            || ("ontouchstart" in window)
            || (navigator.maxTouchPoints && navigator.maxTouchPoints > 0)
        );
    }

    function liberarStreamMicrofono() {
        if (!state.streamMicrofono) return;
        try {
            state.streamMicrofono.getTracks().forEach((t) => {
                try { t.stop(); } catch (_) {}
            });
        } catch (_) {}
        state.streamMicrofono = null;
    }

    async function obtenerStreamMicrofonoValido() {
        if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
            const err = new Error("MIC_REQUIRE_HTTPS");
            err.codigoHaku = "MIC_REQUIRE_HTTPS";
            throw err;
        }

        // Verificar si el stream existente sigue vivo, activo y SIN estar silenciado por el SO móvil
        if (state.streamMicrofono) {
            const tracks = state.streamMicrofono.getAudioTracks();
            const trackValido = (
                tracks.length > 0
                && tracks[0].readyState === "live"
                && tracks[0].enabled
                && !tracks[0].muted
            );
            if (trackValido) {
                return state.streamMicrofono;
            }
            // Si el track feneció, quedó muteado por cambio de pestaña/TTS o está inactivo, re-adquirir
            liberarStreamMicrofono();
        }

        // Solicitar MediaStream con restricciones estándar (evitando forzar sampleRate: 16000 que
        // en Windows/Android silencia o entra en conflicto con drivers nativos a 48000Hz).
        const constraints = {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        };

        const nuevoStream = await navigator.mediaDevices.getUserMedia(constraints);
        const pistas = nuevoStream.getAudioTracks();
        if (pistas.length > 0) {
            const settings = pistas[0].getSettings ? pistas[0].getSettings() : {};
            console.log("[HakuVoz] Micrófono capturado:", pistas[0].label,
                "| readyState:", pistas[0].readyState,
                "| muted:", pistas[0].muted,
                "| sampleRate:", settings.sampleRate,
                "| channels:", settings.channelCount);
        }
        state.streamMicrofono = nuevoStream;
        return nuevoStream;
    }

    async function iniciarGrabacion() {
        if (state.grabando || state.procesando || state.iniciandoGrabacion || state.deteniendoGrabacion) return;

        state.iniciandoGrabacion = true;
        state.detencionPendiente = false;

        try {
            const stream = await obtenerStreamMicrofonoValido();

            // Si el usuario soltó el control antes de que getUserMedia terminara
            if (state.detencionPendiente) {
                state.iniciandoGrabacion = false;
                state.detencionPendiente = false;
                if (esDispositivoTactil()) liberarStreamMicrofono();
                actualizarEstadoVisual("reposo");
                return;
            }

            if (typeof MediaRecorder === "undefined") {
                const errMr = new Error("MEDIARECORDER_UNSUPPORTED");
                errMr.codigoHaku = "MEDIARECORDER_UNSUPPORTED";
                throw errMr;
            }

            // Seleccionar el formato de grabación soportado por el dispositivo cliente
            let mimeType = "";
            if (typeof MediaRecorder.isTypeSupported === "function") {
                for (const t of CONFIG.MIME_TYPES) {
                    if (MediaRecorder.isTypeSupported(t)) {
                        mimeType = t;
                        break;
                    }
                }
            }

            state.audioChunks = [];
            state.mediaRecorder = mimeType
                ? new MediaRecorder(stream, { mimeType })
                : new MediaRecorder(stream);

            state.mediaRecorder.ondataavailable = (event) => {
                const size = (event.data && event.data.size) || 0;
                console.log("[HakuVoz] ondataavailable recibido:", size, "bytes");
                if (size > 0) {
                    state.audioChunks.push(event.data);
                }
            };

            state.mediaRecorder.onerror = (event) => {
                console.error("[HakuVoz] Error de MediaRecorder:", event && event.error ? event.error : event);
                detenerMedidorNivelAudio();
                state.grabando = false;
                state.iniciandoGrabacion = false;
                state.deteniendoGrabacion = false;
                state.procesando = false;
                if (esDispositivoTactil()) liberarStreamMicrofono();
                mostrarToast("Error al grabar audio desde este dispositivo. Intenta de nuevo.", "error");
                actualizarEstadoVisual("reposo");
            };

            state.mediaRecorder.onstop = () => {
                try {
                    const tFinGrabacion = performance.now();
                    const duracionSeg = (Date.now() - state.tiempoInicioGrabacion) / 1000;
                    detenerMedidorNivelAudio();
                    // En móviles (Android/iOS), liberar las pistas del micrófono al detener la grabación
                    // saca al sistema del modo llamada/SCO antes de reproducir la respuesta TTS y evita
                    // tracks silenciados en el siguiente turno.
                    if (esDispositivoTactil()) {
                        liberarStreamMicrofono();
                    }
                    const interactionId = generarIdAleatorio("int");
                    state.interactionActual = {
                        id: interactionId,
                        tFinGrabacion: tFinGrabacion,
                        tInicioComando: null,
                        tTextoMostrado: null,
                        tTTSWavRecibido: null,
                        tPlaying: null,
                    };
                    registrarTimingLog(interactionId, "recording_end", 0, { duracion_grabacion_s: duracionSeg });
                    console.log("[HakuVoz] mediaRecorder onstop. Chunks:", state.audioChunks.length,
                        "| Duración:", duracionSeg.toFixed(2), "s",
                        "| Max RMS entrada:", state.maxRmsGrabacion.toFixed(4));
                    procesarAudioGrabado(duracionSeg, interactionId);
                } catch (errOnstop) {
                    console.error("[HakuVoz] Error en mediaRecorder onstop:", errOnstop);
                    state.deteniendoGrabacion = false;
                    state.procesando = false;
                    actualizarEstadoVisual("reposo");
                }
            };

            state.tiempoInicioGrabacion = Date.now();
            state.grabando = true;
            state.iniciandoGrabacion = false;

            iniciarMedidorNivelAudio(stream);

            // En Safari/iOS con MP4/AAC, evitar timeslice evita contenedores fragmentados sin cabecera limpia.
            const mimeEfectivo = (state.mediaRecorder.mimeType || mimeType || "").toLowerCase();
            if (mimeEfectivo.includes("mp4") || mimeEfectivo.includes("aac")) {
                state.mediaRecorder.start();
            } else {
                state.mediaRecorder.start(250);
            }
            console.log("[HakuVoz] mediaRecorder iniciado con mimeType:", mimeEfectivo || "default");
            actualizarEstadoVisual("grabando");

            // Si se solicitó detención mientras llamábamos a start()
            if (state.detencionPendiente) {
                console.log("[HakuVoz] detención pendiente detectada tras start()");
                state.detencionPendiente = false;
                detenerGrabacion();
            }
        } catch (error) {
            state.iniciandoGrabacion = false;
            state.detencionPendiente = false;
            console.error("[HakuVoz] Error al acceder al micrófono:", error);
            if (error && error.codigoHaku === "MIC_REQUIRE_HTTPS") {
                mostrarToast(
                    "El micrófono requiere conexión segura (HTTPS) desde otro dispositivo. Abre Hacu usando https:// y acepta el certificado.",
                    "error"
                );
            } else if (error && error.codigoHaku === "MEDIARECORDER_UNSUPPORTED") {
                mostrarToast(
                    "Este navegador móvil no soporta grabación directa (MediaRecorder). Usa Chrome, Edge o Safari actualizado.",
                    "error"
                );
            } else {
                mostrarToast("No se pudo acceder al micrófono. Verifica los permisos del navegador o el certificado HTTPS.", "error");
            }
            actualizarEstadoVisual("reposo");
        }
    }

    function detenerGrabacion() {
        console.log("[HakuVoz] detenerGrabacion llamado. grabando:", state.grabando, "iniciando:", state.iniciandoGrabacion, "deteniendo:", state.deteniendoGrabacion);
        if (state.iniciandoGrabacion) {
            state.detencionPendiente = true;
            return;
        }
        if (!state.grabando || !state.mediaRecorder || state.deteniendoGrabacion) return;

        // Guardia atómica contra dobles envíos y múltiples llamadas concurrentes
        state.deteniendoGrabacion = true;
        state.grabando = false;
        actualizarEstadoVisual("procesando");

        if (state.timerPostRoll) {
            clearTimeout(state.timerPostRoll);
            state.timerPostRoll = null;
        }

        // Margen post-roll calibrado a 300 ms para evitar que Windows WASAPI/Chromium
        // recorte la última sílaba o letra final (p. ej. «Almacén A») al hablar rápido
        state.timerPostRoll = setTimeout(() => {
            state.timerPostRoll = null;
            try {
                if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
                    // Al llamar a stop(), el estándar W3C MediaRecorder dispara inmediatamente
                    // el último ondataavailable con los datos restantes y a continuación onstop.
                    // state.deteniendoGrabacion se mantiene activo hasta que onstop/procesarAudioGrabado
                    // capture los datos de forma segura en el audioBlob para evitar carreras con nuevas grabaciones.
                    state.mediaRecorder.stop();
                } else {
                    console.warn("[HakuVoz] Grabador no activo al vencer post-roll.");
                    state.deteniendoGrabacion = false;
                    state.procesando = false;
                    actualizarEstadoVisual("reposo");
                }
            } catch (e) {
                console.error("[HakuVoz] Error al detener grabador tras post-roll:", e);
                state.deteniendoGrabacion = false;
                state.procesando = false;
                actualizarEstadoVisual("reposo");
            }
        }, 300);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // RESAMPLING Y ENCODING DE AUDIO A 16 KHZ MONO (PCM 16-BIT WAV)
    // ─────────────────────────────────────────────────────────────────────────

    function resampleLinear(samples, sourceRate, targetRate) {
        if (sourceRate === targetRate) return samples;
        const ratio = sourceRate / targetRate;
        const targetLength = Math.round(samples.length / ratio);
        const result = new Float32Array(targetLength);
        for (let i = 0; i < targetLength; i++) {
            const srcPos = i * ratio;
            const index = Math.floor(srcPos);
            const fraction = srcPos - index;
            const nextIndex = Math.min(index + 1, samples.length - 1);
            result[i] = samples[index] * (1 - fraction) + samples[nextIndex] * fraction;
        }
        return result;
    }

    function encodeWavPcm16(samples, sampleRate = 16000) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeString(offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        // Encabezado RIFF (44 bytes estándar)
        writeString(0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, "WAVE");

        // Subchunk 'fmt '
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);                          // Formato 1 = PCM
        view.setUint16(22, 1, true);                          // 1 Canal (Mono)
        view.setUint32(24, sampleRate, true);                 // 16000 Hz
        view.setUint32(28, sampleRate * 2, true);             // Byte rate (16000 * 1 * 2 = 32000)
        view.setUint16(32, 2, true);                          // Block align
        view.setUint16(34, 16, true);                         // 16 bits por muestra

        // Subchunk 'data'
        writeString(36, "data");
        view.setUint32(40, samples.length * 2, true);

        // Muestras PCM con saturación [-1.0, 1.0] -> int16
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([buffer], { type: "audio/wav" });
    }

    async function detectarExtensionPorCabecera(audioBlob) {
        try {
            const headBuf = await audioBlob.slice(0, 16).arrayBuffer();
            const b = new Uint8Array(headBuf);
            if (b.length >= 4 && b[0] === 0x1A && b[1] === 0x45 && b[2] === 0xDF && b[3] === 0xA3) {
                return "grabacion.webm";
            }
            if (b.length >= 4 && b[0] === 0x4F && b[1] === 0x67 && b[2] === 0x67 && b[3] === 0x53) {
                return "grabacion.ogg";
            }
            if (b.length >= 4 && b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46) {
                return "grabacion.wav";
            }
            const ascii = String.fromCharCode(...b);
            if (ascii.includes("ftyp")) {
                return "grabacion.mp4";
            }
        } catch (_) {}
        return null;
    }

    async function resamplearAudioA16kMono(audioBlob) {
        // Los contenedores comprimidos de MediaRecorder (WebM/Opus, OGG, MP4/AAC, 3GP, Matroska)
        // se envían directamente al AudioDecoder (PyAV/ffmpeg) del servidor para evitar truncamientos
        // de decodeAudioData() en navegadores de escritorio y móviles (Android/iOS).
        const mime = ((audioBlob && audioBlob.type) || "").toLowerCase();
        if (mime.includes("webm") || mime.includes("matroska")) {
            return { blob: audioBlob, filename: "grabacion.webm" };
        }
        if (mime.includes("ogg") || mime.includes("opus")) {
            return { blob: audioBlob, filename: "grabacion.ogg" };
        }
        if (mime.includes("mp4") || mime.includes("m4a")) {
            return { blob: audioBlob, filename: "grabacion.mp4" };
        }
        if (mime.includes("aac")) {
            return { blob: audioBlob, filename: "grabacion.aac" };
        }
        if (mime.includes("3gpp")) {
            return { blob: audioBlob, filename: "grabacion.3gp" };
        }
        if (mime.includes("mpeg") || mime.includes("mp3")) {
            return { blob: audioBlob, filename: "grabacion.mp3" };
        }
        if (!mime || mime === "application/octet-stream") {
            const extDetectada = await detectarExtensionPorCabecera(audioBlob);
            if (extDetectada && !extDetectada.endsWith(".wav")) {
                return { blob: audioBlob, filename: extDetectada };
            }
        }

        // Si es audio/wav o formato sin clusterizado opaco, procesar en cliente si AudioContext existe
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) {
            return { blob: audioBlob, filename: "grabacion.wav" };
        }

        let audioCtx = null;
        try {
            audioCtx = new AudioContextClass();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await new Promise((resolve, reject) => {
                const result = audioCtx.decodeAudioData(arrayBuffer, resolve, reject);
                if (result && typeof result.then === "function") {
                    result.then(resolve).catch(reject);
                }
            });
            const numChannels = audioBuffer.numberOfChannels;
            const length = audioBuffer.length;
            let monoData = null;
            if (numChannels === 1) {
                monoData = audioBuffer.getChannelData(0);
            } else {
                monoData = new Float32Array(length);
                for (let c = 0; c < numChannels; c++) {
                    const channelData = audioBuffer.getChannelData(c);
                    for (let i = 0; i < length; i++) {
                        monoData[i] += channelData[i] / numChannels;
                    }
                }
            }
            const resampledData = resampleLinear(monoData, audioBuffer.sampleRate, 16000);
            const wavBlob = encodeWavPcm16(resampledData, 16000);
            return { blob: wavBlob, filename: "grabacion.wav" };
        } catch (err) {
            return { blob: audioBlob, filename: "grabacion.wav" };
        } finally {
            if (audioCtx && typeof audioCtx.close === "function") {
                try { audioCtx.close(); } catch (_) {}
            }
        }
    }

    async function procesarAudioGrabado(duracionSeg = 0, interactionIdPropagado = null) {
        if (state.procesando) {
            console.warn("[HakuVoz] procesarAudioGrabado ignorado: ya se encuentra procesando.");
            state.deteniendoGrabacion = false;
            return;
        }

        const interactionId = interactionIdPropagado || (state.interactionActual && state.interactionActual.id) || generarIdAleatorio("int");
        if (!state.interactionActual || state.interactionActual.id !== interactionId) {
            state.interactionActual = {
                id: interactionId,
                tFinGrabacion: performance.now(),
                tInicioComando: null,
                tTextoMostrado: null,
                tTTSWavRecibido: null,
                tPlaying: null,
            };
        }

        console.log("[HakuVoz] procesarAudioGrabado iniciado. Chunks:", state.audioChunks.length,
            "| Duración:", duracionSeg ? duracionSeg.toFixed(2) + "s" : "N/D",
            "| Max RMS entrada:", state.maxRmsGrabacion.toFixed(4));

        if (state.audioChunks.length === 0) {
            console.warn("[HakuVoz] Descartando: state.audioChunks está vacío.");
            state.deteniendoGrabacion = false;
            mostrarToast("Grabación vacía. Mantén pulsada la barra espaciadora mientras hablas.", "info");
            actualizarEstadoVisual("reposo");
            return;
        }

        const chunkMime = (state.audioChunks[0] && state.audioChunks[0].type) || "";
        const mime = (state.mediaRecorder && state.mediaRecorder.mimeType) || chunkMime || "";
        const audioBlob = mime
            ? new Blob(state.audioChunks, { type: mime })
            : new Blob(state.audioChunks);
        console.log("[HakuVoz] audioBlob generado:", audioBlob.size, "bytes, mime:", audioBlob.type);
        state.audioChunks = [];

        // Filtro de integridad: si el buffer carece de datos útiles (< 100 bytes no contiene ni cabecera ni payload).
        if (audioBlob.size < 100) {
            console.warn("[HakuVoz] Descartando: audioBlob sin datos utilizables (", audioBlob.size, "bytes)");
            state.deteniendoGrabacion = false;
            mostrarToast("Grabación sin audio capturado. Intenta hablar nuevamente.", "info");
            actualizarEstadoVisual("reposo");
            return;
        }

        state.procesando = true;
        state.deteniendoGrabacion = false;
        actualizarEstadoVisual("procesando");

        // Alerta preventiva no bloqueante si el hardware/navegador entregó señal cero (micrófono silenciado)
        if (state.maxRmsGrabacion < 0.005) {
            console.warn("[HakuVoz] ADVERTENCIA: La amplitud de entrada capturada por el navegador fue prácticamente nula (Max RMS = " + state.maxRmsGrabacion.toFixed(5) + "). Comprueba que el micrófono no esté muteado en Windows o con volumen en cero.");
        }

        try {
            reiniciarTimerInactividad();

            // Preparar audio con resampling 16kHz mono (o fallback transparente a Blob original)
            const t0Resample = performance.now();
            const { blob: audioParaEnviar, filename } = await resamplearAudioA16kMono(audioBlob);
            const dtResample = performance.now() - t0Resample;
            registrarTimingLog(interactionId, "browser_resample", dtResample, { bytes: audioParaEnviar.size });

            // Enviar audio al backend (AudioDecoder lo procesa a 16kHz mono)
            const reqIdSTT = generarIdAleatorio("req");
            const formData = new FormData();
            formData.append("audio", audioParaEnviar, filename);

            const tInicioSTT = performance.now();
            registrarTimingLog(interactionId, "stt_send", 0, { request_id: reqIdSTT });

            const timeoutSTT = crearSenalTimeout(CONFIG.TIMEOUT_STT_MS);
            let respSTT;
            try {
                respSTT = await apiFetch(CONFIG.ENDPOINTS.TRANSCRIBIR, {
                    method: "POST",
                    headers: {
                        "X-Haku-Interaction-ID": interactionId,
                        "X-Haku-Request-ID": reqIdSTT,
                    },
                    body: formData,
                    signal: timeoutSTT.signal,
                });
            } finally {
                timeoutSTT.limpiar();
            }

            const tFinSTT = performance.now();
            const dtSTT = tFinSTT - tInicioSTT;
            registrarTimingLog(interactionId, "stt_received", dtSTT, { request_id: reqIdSTT });

            let dataSTT = {};
            try {
                dataSTT = await respSTT.json();
            } catch (_) {
                dataSTT = {};
            }
            console.log("[HakuVoz] Respuesta STT recibida:", respSTT.status, dataSTT);

            // Manejo explícito de 503 (motor de voz aún cargando o lock ocupado)
            if (
                respSTT.status === 503
                || dataSTT.status === "motor_cargando"
                || dataSTT.estado === "MOTOR_CARGANDO"
                || dataSTT.estado === "HAKU_OCUPADO"
            ) {
                const motivoEspera = (dataSTT.estado === "MOTOR_CARGANDO" || dataSTT.status === "motor_cargando")
                    ? "stt_loading"
                    : "stt_busy";
                registrarTimingLog(interactionId, motivoEspera, dtSTT, { request_id: reqIdSTT, status: respSTT.status });
                const mensajeEspera = dataSTT.mensaje_pantalla
                    || (motivoEspera === "stt_loading"
                        ? "El motor de voz aún está terminando de iniciar. Espera unos segundos e intenta de nuevo."
                        : "Hacu Inventario está atendiendo otra solicitud. Intenta nuevamente en unos instantes.");
                mostrarToast(mensajeEspera, "info");
                actualizarEstadoVisual("reposo");
                state.procesando = false;
                return;
            }

            // Distinguir el audio ilegible del silencio: el usuario puede hablar
            // perfectamente y aun así fallar la lectura de un contenedor/códec.
            if (dataSTT.status === "error_audio" || dataSTT.estado === "ERROR_AUDIO") {
                registrarTimingLog(interactionId, "stt_audio_decode_error", dtSTT, { request_id: reqIdSTT });
                mostrarToast(dataSTT.mensaje_pantalla || "No pude leer el formato de audio recibido. Intenta grabar otra vez.", "error");
                actualizarEstadoVisual("reposo");
                state.procesando = false;
                return;
            }

            // 2. Manejo explícito de silencio VAD
            if (dataSTT.status === "silencio" || dataSTT.estado === "SIN_VOZ") {
                registrarTimingLog(interactionId, "stt_silence", dtSTT);
                let mensajeAviso = dataSTT.mensaje_pantalla || "No se detectó voz audible en el micrófono.";
                if (state.maxRmsGrabacion < 0.005) {
                    mensajeAviso += " (El micrófono parece no recibir señal física o estar silenciado en Windows).";
                }
                mostrarToast(mensajeAviso, "info");
                actualizarEstadoVisual("reposo");
                state.procesando = false;
                return;
            }

            // 3. Manejo explícito de error en el motor de transcripción STT o HTTP 4xx/5xx no manejado
            if (
                !respSTT.ok
                || dataSTT.status === "error_transcripcion"
                || dataSTT.estado === "ERROR_TRANSCRIPCION"
            ) {
                registrarTimingLog(interactionId, "stt_error", dtSTT, { status: respSTT.status });
                const mensajeError = dataSTT.mensaje_pantalla
                    || dataSTT.error
                    || `Hubo un problema al transcribir el audio en el servidor (HTTP ${respSTT.status}).`;
                mostrarToast(mensajeError, "error");
                actualizarEstadoVisual("reposo");
                state.procesando = false;
                return;
            }

            const textoTranscrito = (dataSTT.texto || "").trim();

            if (!textoTranscrito) {
                mostrarToast("No se detectó voz clara. Intenta nuevamente.", "info");
                actualizarEstadoVisual("reposo");
                state.procesando = false;
                return;
            }

            // Mostrar lo que el usuario dijo
            agregarMensajeChat("usuario", textoTranscrito);

            // 3. Procesar el comando con Haku (se transfiere el flujo liberando la etapa STT)
            state.procesando = false;
            await enviarComandoHaku(textoTranscrito, interactionId);

        } catch (error) {
            console.error("Error procesando audio:", error);
            registrarTimingLog(interactionId, esErrorTimeout(error) ? "stt_timeout" : "stt_exception", 0, { error: error?.message || String(error) });
            mostrarToast(
                esErrorTimeout(error)
                    ? "La transcripción tardó demasiado en responder. Intenta grabar nuevamente."
                    : "Error de conexión al procesar audio.",
                "error"
            );
            actualizarEstadoVisual("reposo");
        } finally {
            state.procesando = false;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ENVÍO DE COMANDO Y RESPUESTA
    // ─────────────────────────────────────────────────────────────────────────

    async function enviarComandoHaku(mensajeTexto, interactionIdPropagado = null) {
        if (state.procesando) {
            console.warn("[HakuVoz] Solicitud ignorada: ya hay un comando en proceso.");
            return;
        }
        state.procesando = true;
        const btnEnviar = document.getElementById("haku-btn-enviar");
        if (btnEnviar) btnEnviar.disabled = true;

        reiniciarTimerInactividad();
        actualizarEstadoVisual("procesando");

        const interactionId = interactionIdPropagado || (state.interactionActual && state.interactionActual.id) || generarIdAleatorio("int");
        if (!state.interactionActual || state.interactionActual.id !== interactionId) {
            state.interactionActual = {
                id: interactionId,
                tFinGrabacion: null,
                tInicioComando: performance.now(),
                tTextoMostrado: null,
                tTTSWavRecibido: null,
                tPlaying: null,
            };
        }

        const reqIdMsg = generarIdAleatorio("req");
        const tInicioMensaje = performance.now();
        registrarTimingLog(interactionId, "message_send", 0, { request_id: reqIdMsg });

        try {
            const payloadMensaje = {
                mensaje: mensajeTexto,
                conversacion_id: state.conversacionId,
            };
            const inpObsDevMsg = document.getElementById("haku-dev-observacion");
            if (inpObsDevMsg) {
                payloadMensaje.observacion = inpObsDevMsg.value.trim();
            }
            const timeoutMsg = crearSenalTimeout(CONFIG.TIMEOUT_MENSAJE_MS);
            let resp;
            try {
                resp = await apiFetch(CONFIG.ENDPOINTS.MENSAJE, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-Haku-Interaction-ID": interactionId,
                        "X-Haku-Request-ID": reqIdMsg,
                    },
                    body: JSON.stringify(payloadMensaje),
                    signal: timeoutMsg.signal,
                });
            } finally {
                timeoutMsg.limpiar();
            }

            const tFinMensaje = performance.now();
            const dtMsgHttp = tFinMensaje - tInicioMensaje;
            registrarTimingLog(interactionId, "message_received", dtMsgHttp, { request_id: reqIdMsg });

            let data = {};
            try {
                data = await resp.json();
            } catch (_) {
                data = {};
            }

            if (data.error || !resp.ok) {
                if (data.propuesta_activa === false) {
                    state.propuestaActiva = null;
                    ocultarTarjetaPropuesta();
                }
                const msgErrMsg = data.mensaje_pantalla
                    || data.mensaje
                    || data.error
                    || `El servidor devolvió un error inesperado (HTTP ${resp.status}).`;
                agregarMensajeChat("haku", `⚠️ **Error:** ${msgErrMsg}`);
                if (data.mensaje_verbal) reproducirVozCliente(data.mensaje_verbal, interactionId);
                actualizarEstadoVisual("reposo");
                return;
            }

            // Si es una propuesta pendiente de confirmación
            const requiereEscDest = Boolean(data.requiere_escritura || (data.datos && data.datos.requiere_escritura));
            const esperandoDest = data.esperando || (data.datos && data.datos.esperando);
            if (data.tipo === "propuesta" && data.propuesta) {
                state.propuestaActiva = data.propuesta;
                mostrarTarjetaPropuesta(data.propuesta);
            } else if (data.tipo === "aclaracion" && requiereEscDest && esperandoDest === "destinatario") {
                state.propuestaActiva = null;
                mostrarFormularioDestinatario({
                    mensaje_pantalla: data.mensaje_pantalla,
                    candidatos: data.candidatos || (data.datos && data.datos.candidatos) || [],
                });
            } else if (data.tipo === "formulario_crear_producto" && data.datos) {
                state.propuestaActiva = null;
                mostrarFormularioCrearProducto(data.datos);
            } else if (data.tipo === "aclaracion" && data.datos && data.datos.operacion_pendiente === "CREAR_PRODUCTO") {
                state.propuestaActiva = null;
                const catalogoCrear = obtenerDatosCatalogoCrearProducto();
                const prefill = Object.assign({ _conversacional: true }, data.datos.datos_producto || {});
                mostrarFormularioCrearProducto(catalogoCrear, prefill);
            } else if (data.tipo === "formulario_editar_producto" && data.datos) {
                state.propuestaActiva = null;
                mostrarFormularioEditarProducto(data.datos);
            } else if (data.tipo === "formulario_crear_usuario" && data.datos) {
                state.propuestaActiva = null;
                mostrarFormularioCrearUsuario(data.datos);
            } else if (data.tipo === "formulario_editar_usuario" && data.datos) {
                state.propuestaActiva = null;
                mostrarFormularioEditarUsuario(data.datos);
            } else if (
                data.tipo === "confirmacion_ejecutada"
                || data.tipo === "propuesta_cancelada"
                || data.tipo === "accion_cancelar"
                || (data.propuesta_activa === false && state.propuestaActiva)
            ) {
                state.propuestaActiva = null;
                ocultarTarjetaPropuesta();
            }

            // Mostrar respuesta en chat
            if (data.mensaje_pantalla) {
                agregarMensajeChat("haku", data.mensaje_pantalla, data.tipo, data.datos);
                const tTextoMostrado = performance.now();
                if (state.interactionActual && state.interactionActual.id === interactionId) {
                    state.interactionActual.tTextoMostrado = tTextoMostrado;
                }
                const baseT = state.interactionActual?.tFinGrabacion || state.interactionActual?.tInicioComando || tInicioMensaje;
                const dtHastaTexto = tTextoMostrado - baseT;
                registrarTimingLog(interactionId, "text_rendered", dtHastaTexto, {
                    desde: state.interactionActual?.tFinGrabacion ? "fin_grabacion" : "envio_texto"
                });
            }

            // Reproducir síntesis de voz en el dispositivo cliente
            if (data.mensaje_verbal) {
                await reproducirVozCliente(data.mensaje_verbal, interactionId);
            } else {
                actualizarEstadoVisual("reposo");
            }

        } catch (error) {
            console.error("Error al enviar comando:", error);
            registrarTimingLog(interactionId, esErrorTimeout(error) ? "message_timeout" : "message_exception", 0, { error: error?.message || String(error) });
            agregarMensajeChat(
                "haku",
                esErrorTimeout(error)
                    ? "⚠️ La consulta tardó demasiado en responder. Intenta nuevamente."
                    : "⚠️ Ocurrió un error al procesar tu solicitud."
            );
            actualizarEstadoVisual("reposo");
        } finally {
            state.procesando = false;
            if (btnEnviar) btnEnviar.disabled = false;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SÍNTESIS DE VOZ Y REPRODUCCIÓN EN CLIENTE (PIPER TTS)
    // ─────────────────────────────────────────────────────────────────────────

    function detenerReproduccionVoz() {
        state.ttsPlaybackId = (state.ttsPlaybackId || 0) + 1;

        if (state.abortControllerTTS) {
            try {
                state.abortControllerTTS.abort();
            } catch (err) {
                console.warn("Error al abortar síntesis TTS:", err);
            }
            state.abortControllerTTS = null;
        }

        if (state.audioElement) {
            try {
                state.audioElement.onended = null;
                state.audioElement.onerror = null;
                state.audioElement.pause();
                state.audioElement.currentTime = 0;
                if (state.audioElement.src && state.audioElement.src.startsWith("blob:")) {
                    URL.revokeObjectURL(state.audioElement.src);
                }
            } catch (err) {
                console.warn("Error al detener elemento de audio:", err);
            }
            state.audioElement = null;
        }

        state.reproduciendoAudio = false;
        actualizarEstadoVisual("reposo");
    }

    async function reproducirVozCliente(textoParaVoz, interactionIdPropagado = null) {
        if (!textoParaVoz) {
            actualizarEstadoVisual("reposo");
            return;
        }

        const interactionId = interactionIdPropagado || (state.interactionActual && state.interactionActual.id) || generarIdAleatorio("int");

        // Cancelar limpiamente cualquier reproducción anterior
        detenerReproduccionVoz();

        const currentPlaybackId = ++state.ttsPlaybackId;

        try {
            actualizarEstadoVisual("hablando");
            state.reproduciendoAudio = true;

            const controller = new AbortController();
            state.abortControllerTTS = controller;

            const reqIdTTS = generarIdAleatorio("req");
            const tInicioTTS = performance.now();
            registrarTimingLog(interactionId, "tts_requested", 0, { request_id: reqIdTTS });

            const resp = await apiFetch(CONFIG.ENDPOINTS.SINTETIZAR, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Haku-Interaction-ID": interactionId,
                    "X-Haku-Request-ID": reqIdTTS,
                },
                body: JSON.stringify({ texto: textoParaVoz }),
                signal: controller.signal,
            });

            if (currentPlaybackId !== state.ttsPlaybackId || controller.signal.aborted || !state.reproduciendoAudio) {
                return;
            }

            if (!resp.ok) {
                registrarTimingLog(interactionId, "tts_error", performance.now() - tInicioTTS, { http_status: resp.status });
                detenerReproduccionVoz();
                return;
            }

            const tFinTTSHttp = performance.now();
            const dtTTSHttp = tFinTTSHttp - tInicioTTS;
            registrarTimingLog(interactionId, "tts_http_client", dtTTSHttp, { request_id: reqIdTTS });

            // 9. Duración de response.blob() por separado
            const t0Blob = performance.now();
            const audioBlob = await resp.blob();
            const dtBlob = performance.now() - t0Blob;
            const tBlobFin = performance.now();
            registrarTimingLog(interactionId, "blob_parsed", dtBlob, { wav_bytes: audioBlob.size });

            if (state.interactionActual && state.interactionActual.id === interactionId) {
                state.interactionActual.tTTSWavRecibido = tBlobFin;
            }

            if (currentPlaybackId !== state.ttsPlaybackId || controller.signal.aborted || !state.reproduciendoAudio) {
                return;
            }

            // 10. Creación del objeto de audio
            const t0Audio = performance.now();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            state.audioElement = audio;
            const dtAudioCreacion = performance.now() - t0Audio;
            registrarTimingLog(interactionId, "audio_created", dtAudioCreacion);

            // 11. Evento canplay y reproducción efectiva (playing)
            audio.addEventListener("canplay", () => {
                const dtCanplay = performance.now() - tBlobFin;
                registrarTimingLog(interactionId, "audio_canplay", dtCanplay);
            }, { once: true });

            audio.addEventListener("playing", () => {
                const tPlaying = performance.now();
                const dtWavToPlaying = tPlaying - tBlobFin;
                const dtTotalE2E = state.interactionActual?.tFinGrabacion ? (tPlaying - state.interactionActual.tFinGrabacion) : null;
                const dtTotalDesdeTexto = state.interactionActual?.tInicioComando ? (tPlaying - state.interactionActual.tInicioComando) : null;
                const duracionE2E = dtTotalE2E || dtTotalDesdeTexto || (tPlaying - tInicioTTS);

                registrarTimingLog(interactionId, "audio_playing", dtWavToPlaying, {
                    total_e2e_ms: duracionE2E,
                    desde_fin_grabacion: Boolean(dtTotalE2E),
                });

                // Resumen final de la interacción
                const tHastaTexto = (state.interactionActual && state.interactionActual.tTextoMostrado && state.interactionActual.tFinGrabacion)
                    ? (state.interactionActual.tTextoMostrado - state.interactionActual.tFinGrabacion)
                    : null;

                console.log(
                    `[HAKU-TIMING-SUMMARY] interaction=${interactionId}` +
                    (tHastaTexto !== null ? ` tiempo_hasta_texto_ms=${tHastaTexto.toFixed(2)}` : "") +
                    ` tts_http_total_ms=${dtTTSHttp.toFixed(2)}` +
                    ` wav_hasta_playing_ms=${dtWavToPlaying.toFixed(2)}` +
                    ` total_e2e_audible_ms=${duracionE2E.toFixed(2)}`
                );
            }, { once: true });

            // 12. Errores de red, decodificación o reproducción
            audio.addEventListener("error", (e) => {
                const errCode = audio.error ? audio.error.code : "unknown";
                const errMsg = audio.error ? audio.error.message : "";
                registrarTimingLog(interactionId, "audio_error", 0, { code: errCode, msg: errMsg });
            }, { once: true });

            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                if (state.ttsPlaybackId === currentPlaybackId) {
                    state.audioElement = null;
                    state.reproduciendoAudio = false;
                    actualizarEstadoVisual("reposo");
                }
            };

            audio.onerror = () => {
                URL.revokeObjectURL(audioUrl);
                if (state.ttsPlaybackId === currentPlaybackId) {
                    state.audioElement = null;
                    state.reproduciendoAudio = false;
                    actualizarEstadoVisual("reposo");
                }
            };

            if (currentPlaybackId !== state.ttsPlaybackId || controller.signal.aborted || !state.reproduciendoAudio) {
                URL.revokeObjectURL(audioUrl);
                return;
            }

            await audio.play();

        } catch (error) {
            if (error && error.name === "AbortError") {
                // Interrupción intencional del usuario (barge-in); silenciar
                return;
            }
            registrarTimingLog(interactionId, "audio_playback_exception", 0, { error: error?.message || String(error) });
            console.warn("No se pudo reproducir audio en cliente:", error);
            detenerReproduccionVoz();
        }
    }


    // ─────────────────────────────────────────────────────────────────────────
    // PRE-CONFIRMACIÓN: CONFIRMAR O CANCELAR PROPUESTA
    // ─────────────────────────────────────────────────────────────────────────

    async function confirmarPropuestaManual(propuestaId) {
        if (!propuestaId || state.procesando) return;
        state.procesando = true;
        reiniciarTimerInactividad();
        actualizarEstadoVisual("procesando");

        try {
            const inpObsDev = document.getElementById("haku-dev-observacion");
            const observacionVal = inpObsDev ? inpObsDev.value.trim() : "";
            const bodyConfirm = {
                propuesta_id: propuestaId,
                conversacion_id: state.conversacionId,
            };
            if (inpObsDev) {
                bodyConfirm.observacion = observacionVal;
            }
            const resp = await apiFetch(CONFIG.ENDPOINTS.CONFIRMAR, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(bodyConfirm),
            });

            const data = await resp.json();

            if (data.error) {
                const erroresTerminales = new Set([
                    "propuesta_cancelada",
                    "propuesta_sustituida",
                    "propuesta_ya_ejecutada",
                    "propuesta_contrato_anterior",
                    "propuesta_no_activa",
                    "propuesta_caducada",
                ]);
                if (resp.status === 404 || data.propuesta_activa === false || erroresTerminales.has(String(data.error))) {
                    state.propuestaActiva = null;
                    ocultarTarjetaPropuesta();
                }
                const msgErr = data.mensaje_pantalla || data.mensaje || data.error;
                mostrarToast(msgErr, "error");
                agregarMensajeChat("haku", `⚠️ No se pudo completar: ${msgErr}`);
                if (data.mensaje_verbal) {
                    await reproducirVozCliente(data.mensaje_verbal);
                }
            } else {
                state.propuestaActiva = null;
                ocultarTarjetaPropuesta();
                agregarMensajeChat("haku", `✅ ${data.mensaje_pantalla}`, "confirmacion_ejecutada");
                if (data.mensaje_verbal) {
                    await reproducirVozCliente(data.mensaje_verbal);
                }
            }
        } catch (error) {
            console.error("Error al confirmar propuesta:", error);
            mostrarToast("Error al confirmar la transacción.", "error");
        } finally {
            state.procesando = false;
            actualizarEstadoVisual("reposo");
        }
    }

    async function cancelarPropuestaManual(propuestaId) {
        if (!propuestaId || state.procesando) return;
        state.procesando = true;
        reiniciarTimerInactividad();
        actualizarEstadoVisual("procesando");

        try {
            const resp = await apiFetch(CONFIG.ENDPOINTS.CANCELAR, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    propuesta_id: propuestaId,
                    conversacion_id: state.conversacionId,
                }),
            });
            const data = await resp.json().catch(() => ({}));
            if (data.error && resp.status >= 400 && resp.status !== 404) {
                mostrarToast(data.mensaje || data.error, "error");
                agregarMensajeChat("haku", `⚠️ ${data.mensaje || data.error}`);
                return;
            }

            state.propuestaActiva = null;
            ocultarTarjetaPropuesta();
            agregarMensajeChat("haku", "❌ Operación cancelada por el usuario.", "propuesta_cancelada");
            await reproducirVozCliente("Operación cancelada. No se realizaron cambios.");
        } catch (error) {
            console.error("Error al cancelar propuesta:", error);
            mostrarToast("Error de conexión al cancelar la propuesta.", "error");
        } finally {
            state.procesando = false;
            actualizarEstadoVisual("reposo");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PUSH-TO-TALK CON BARRA ESPACIADORA (PC)
    // ─────────────────────────────────────────────────────────────────────────

    function configurarPushToTalkTeclado() {
        window.addEventListener("keydown", (e) => {
            // Verificar si es la barra espaciadora
            if (e.code !== "Space" && e.key !== " ") return;

            // Ignorar repeticiones continuas automáticas mientras la tecla se mantiene pulsada
            if (e.repeat) return;

            // No interceptar si el usuario está escribiendo en un input o editable
            const activo = document.activeElement;
            const esCampoDeTexto =
                activo &&
                (activo.tagName === "INPUT" ||
                 activo.tagName === "TEXTAREA" ||
                 activo.tagName === "SELECT" ||
                 activo.isContentEditable);

            if (esCampoDeTexto) return;

            e.preventDefault(); // Evitar scroll de la página con espacio

            // 1. REGLA BARGE-IN: Si HAKU está reproduciendo voz, la pulsación cancela inmediatamente
            // la reproducción y se consume como interrupción (NO debe iniciar grabación accidental).
            if (state.reproduciendoAudio) {
                detenerReproduccionVoz();
                state.interrupcionConsumida = true;
                return;
            }

            if (!state.espacioPresionado && !state.grabando && !state.procesando && !state.deteniendoGrabacion) {
                state.espacioPresionado = true;
                iniciarGrabacion();
            }
        });

        window.addEventListener("keyup", (e) => {
            if (e.code !== "Space" && e.key !== " ") return;

            // Si esta pulsación consumió la interrupción de audio, resetear bandera y no iniciar/detener grabación
            if (state.interrupcionConsumida) {
                state.interrupcionConsumida = false;
                return;
            }

            if (state.espacioPresionado) {
                state.espacioPresionado = false;
                detenerGrabacion();
            }
        });

        // Cancelación o cierre seguro al perder el foco de la ventana
        window.addEventListener("blur", () => {
            if (state.espacioPresionado || state.grabando) {
                state.espacioPresionado = false;
                detenerGrabacion();
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // BOTÓN MÓVIL / TÁCTIL (CLICK Y TOUCH)
    // ─────────────────────────────────────────────────────────────────────────

    function configurarBotonMicrofono() {
        const btnMic = document.getElementById("haku-btn-mic");
        if (!btnMic) return;

        let fueTouch = false;
        let touchInicioMs = 0;
        let estabaGrabandoAlTocar = false;
        const UMBRAL_HOLD_MS = 400;

        // Soporte táctil móvil híbrido:
        // - Toque corto (< 400 ms): inicia grabación y queda escuchando hasta tocar de nuevo para enviar.
        // - Pulsar sostenido (>= 400 ms): modo Push-to-Talk, envía automáticamente al soltar.
        btnMic.addEventListener("touchstart", (e) => {
            e.preventDefault();
            fueTouch = true;
            touchInicioMs = Date.now();
            if (state.reproduciendoAudio) {
                detenerReproduccionVoz();
                estabaGrabandoAlTocar = true;
                return;
            }
            estabaGrabandoAlTocar = Boolean(state.grabando || state.iniciandoGrabacion);
            if (estabaGrabandoAlTocar) {
                detenerGrabacion();
            } else {
                iniciarGrabacion();
            }
        }, { passive: false });

        btnMic.addEventListener("touchend", (e) => {
            e.preventDefault();
            const duracionToqueMs = Date.now() - touchInicioMs;
            setTimeout(() => { fueTouch = false; }, 450);
            // Si fue una pulsación sostenida (>= 400 ms), detener al soltar (Push-to-Talk).
            // Si fue un toque breve (< 400 ms), mantener la grabación abierta hasta el siguiente toque.
            if (!estabaGrabandoAlTocar && duracionToqueMs >= UMBRAL_HOLD_MS) {
                if (state.grabando || state.iniciandoGrabacion) {
                    detenerGrabacion();
                }
            }
        }, { passive: false });

        btnMic.addEventListener("touchcancel", () => {
            setTimeout(() => { fueTouch = false; }, 450);
            if (!estabaGrabandoAlTocar && (Date.now() - touchInicioMs) >= UMBRAL_HOLD_MS) {
                if (state.grabando || state.iniciandoGrabacion) {
                    detenerGrabacion();
                }
            }
        });

        // Soporte clic (toggle o interrupción)
        btnMic.addEventListener("click", (e) => {
            e.preventDefault();
            if (fueTouch) return;
            if (state.reproduciendoAudio) {
                detenerReproduccionVoz();
                return;
            }
            if (state.grabando || state.iniciandoGrabacion) {
                detenerGrabacion();
            } else {
                iniciarGrabacion();
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // COMPONENTES DE INTERFAZ GRÁFICA
    // ─────────────────────────────────────────────────────────────────────────

    function actualizarEstadoVisual(nuevoEstado) {
        const btnMic = document.getElementById("haku-btn-mic");
        const statusLabel = document.getElementById("haku-status-label");
        const waveContainer = document.getElementById("haku-wave-container");

        if (btnMic) {
            btnMic.classList.remove("recording", "processing", "speaking");
            if (nuevoEstado === "grabando") btnMic.classList.add("recording");
            if (nuevoEstado === "procesando") btnMic.classList.add("processing");
            if (nuevoEstado === "hablando") btnMic.classList.add("speaking");
        }

        if (statusLabel) {
            switch (nuevoEstado) {
                case "grabando":
                    statusLabel.textContent = "Escuchando... toca el micrófono para enviar (o suelta Espacio)";
                    statusLabel.className = "status-recording";
                    break;
                case "procesando":
                    statusLabel.textContent = "Hacu Inventario está pensando...";
                    statusLabel.className = "status-processing";
                    break;
                case "hablando":
                    statusLabel.textContent = "Hacu Inventario respondiendo...";
                    statusLabel.className = "status-speaking";
                    break;
                default:
                    statusLabel.textContent = "Presiona Espacio o toca el micrófono para hablar";
                    statusLabel.className = "status-idle";
            }
        }

        if (waveContainer) {
            waveContainer.style.display = nuevoEstado === "grabando" || nuevoEstado === "hablando" ? "flex" : "none";
        }
    }

    function construirUrlProducto(codigo, contextoNav = "productos") {
        const q = encodeURIComponent(String(codigo || "").trim());
        if (contextoNav === "prestamos") {
            return `/inventario/movimientos/?vista=PRESTAMOS&q=${q}`;
        }
        if (contextoNav === "alertas_critica") {
            return `/inventario/alertas/?nivel=CRITICA&q=${q}`;
        }
        if (contextoNav === "alertas_baja") {
            return `/inventario/alertas/?nivel=BAJA&q=${q}`;
        }
        if (contextoNav === "alertas") {
            return `/inventario/alertas/?q=${q}`;
        }
        return `/inventario/productos/?q=${q}`;
    }

    function construirUrlMovimiento(codigo, tipoTexto, fechaTexto, contextoNav = "movimientos") {
        const tipoLow = String(tipoTexto || "").toLowerCase();
        let vista = contextoNav === "prestamos" ? "PRESTAMOS" : "TODOS";
        if (tipoLow.includes("préstamo") || tipoLow.includes("prestamo")) {
            vista = "PRESTAMOS";
        } else if (tipoLow.includes("ajuste") || tipoLow.includes("conteo")) {
            vista = "CONTEOS";
        } else if (tipoLow.includes("entrada")) {
            vista = "ENTRADA";
        } else if (tipoLow.includes("traslado")) {
            vista = "TRASLADO";
        } else if (
            tipoLow.includes("devolución") ||
            tipoLow.includes("devolucion") ||
            tipoLow.includes("dotación") ||
            tipoLow.includes("dotacion") ||
            tipoLow.includes("salida") ||
            tipoLow.includes("baja")
        ) {
            vista = "SALIDA";
        }

        const params = [`vista=${vista}`];
        if (codigo) {
            params.push(`q=${encodeURIComponent(String(codigo).trim())}`);
        }
        const mFecha = String(fechaTexto || "").match(/^(\d{2})\/(\d{2})\/(\d{4})/);
        if (mFecha) {
            const iso = `${mFecha[3]}-${mFecha[2]}-${mFecha[1]}`;
            params.push(`fecha_desde=${iso}`, `fecha_hasta=${iso}`);
        }
        return `/inventario/movimientos/?${params.join("&")}`;
    }

    function formatearMarkdownInline(lineaEscapada) {
        return lineaEscapada
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    function formatearMensajeHakuConEnlaces(texto, datos) {
        const contextoNav = (datos && datos.contexto_navegacion) || "productos";
        const urlVerTodos = (datos && datos.url_ver_todos) || (
            contextoNav === "prestamos"
                ? "/inventario/movimientos/?vista=PRESTAMOS"
                : contextoNav === "movimientos"
                    ? "/inventario/movimientos/"
                    : "/inventario/productos/"
        );
        const etiquetaVerTodos = (datos && datos.etiqueta_ver_todos) || (
            contextoNav === "prestamos"
                ? "Mostrar todos los préstamos"
                : contextoNav === "movimientos"
                    ? "Mostrar todos los movimientos"
                    : "Mostrar todos los productos"
        );
        const totalRegistros = datos && typeof datos.total === "number" ? datos.total : null;
        const codigoProductoUnico =
            datos && Array.isArray(datos.productos) && datos.productos.length === 1
                ? datos.productos[0].codigo
                : null;

        const lineas = String(texto || "").split("\n");
        let tieneItemsLista = false;
        let mostroBotonVerTodos = false;

        const lineasHtml = lineas.map((lineaCruda) => {
            const esc = escapeHtml(lineaCruda);

            // 1. Pie de paginación: _Mostrando X de Y resultados._ / _Mostrando X de Y movimientos._
            const mPag = esc.match(/^\s*_Mostrando\s+(\d+)\s+de\s+(\d+)\s+(resultados?|productos?|movimientos?)\._\s*$/i);
            if (mPag) {
                mostroBotonVerTodos = true;
                const totalTxt = mPag[2];
                return (
                    `<div class="haku-ver-todos-wrapper">` +
                    `<em style="font-size:0.82rem; color:var(--text-muted);">Mostrando ${mPag[1]} de ${totalTxt} ${mPag[3]}.</em>` +
                    `<a href="${escapeHtml(urlVerTodos)}" target="_blank" rel="noopener noreferrer" class="haku-ver-todos-btn">` +
                    `${escapeHtml(etiquetaVerTodos)} (${totalTxt}) &rarr;</a>` +
                    `</div>`
                );
            }

            // 2. Fila de movimiento global o de ficha: - dd/mm/YYYY HH:MM · ...
            const mMov = esc.match(/^\s*-\s+(\d{2}\/\d{2}\/\d{4}(?:\s+\d{2}:\d{2})?)\s+·\s+(.+)$/);
            if (mMov) {
                tieneItemsLista = true;
                const fechaTxt = mMov[1];
                const resto = mMov[2];
                const mCod = resto.match(/^\*\*([A-Z0-9]+-[A-Z0-9-]+)\*\*\s+/i);
                const codigoMov = mCod ? mCod[1] : codigoProductoUnico;
                const urlMov = construirUrlMovimiento(codigoMov, resto, fechaTxt, contextoNav);
                const contenido = formatearMarkdownInline(esc.replace(/^\s*-\s+/, ""));
                return (
                    `<a href="${escapeHtml(urlMov)}" target="_blank" rel="noopener noreferrer" class="haku-item-link" title="Abrir este movimiento en la pestaña de Movimientos">` +
                    `<span class="haku-item-text">&bull; ${contenido}</span>` +
                    `<span class="haku-item-arrow" aria-hidden="true">&nearr;</span></a>`
                );
            }

            // 3. Fila de producto o préstamo en lista: - **COD-001** Nombre... (sin dos puntos de atributo como - **Categoría:**)
            const mProd = esc.match(/^\s*-\s+\*\*([A-Z0-9]+-[A-Z0-9-]+)\*\*(?!\s*:)\s+(.+)$/i);
            if (mProd) {
                tieneItemsLista = true;
                const codigo = mProd[1];
                const detalleLinea = mProd[2];
                const esPrestamoLinea =
                    contextoNav === "prestamos" ||
                    /en\s+pr[eé]stamo/i.test(detalleLinea);
                const contextoItem = esPrestamoLinea ? "prestamos" : contextoNav;
                const urlItem = construirUrlProducto(codigo, contextoItem);
                const titulo = esPrestamoLinea
                    ? `Ver préstamo de ${codigo} en el apartado de Préstamos`
                    : contextoItem.startsWith("alertas")
                        ? `Ver alerta de stock de ${codigo} en Alertas de Stock`
                        : `Ver producto ${codigo} en la lista de Catálogo`;
                const contenido = formatearMarkdownInline(esc.replace(/^\s*-\s+/, ""));
                return (
                    `<a href="${escapeHtml(urlItem)}" target="_blank" rel="noopener noreferrer" class="haku-item-link" title="${escapeHtml(titulo)}">` +
                    `<span class="haku-item-text">&bull; ${contenido}</span>` +
                    `<span class="haku-item-arrow" aria-hidden="true">&nearr;</span></a>`
                );
            }

            // 4. Encabezado de ficha ### Nombre (`COD-001`) o mención **Nombre (COD-001)**
            const mHeading = esc.match(/^\s*###\s+(.+)$/);
            if (mHeading) {
                const tituloFmt = formatearMarkdownInline(mHeading[1].trim());
                return `<h4 class="haku-msg-heading">${tituloFmt}</h4>`;
            }
            let lineaFormateada = esc.replace(
                /\*\*([^*]+?\(([A-Z0-9]+-[A-Z0-9-]+)\))\*\*/gi,
                (_, textoCompleto, cod) => {
                    const urlInline = construirUrlProducto(cod, contextoNav);
                    return `<a href="${escapeHtml(urlInline)}" target="_blank" rel="noopener noreferrer" class="haku-inline-link"><strong>${textoCompleto}</strong></a>`;
                }
            );
            lineaFormateada = formatearMarkdownInline(lineaFormateada);
            return lineaFormateada;
        });

        let htmlFinal = lineasHtml.join("<br>");
        // Limpiar <br> redundantes alrededor de bloques .haku-item-link y .haku-ver-todos-wrapper
        htmlFinal = htmlFinal
            .replace(/<\/a><br><a /g, "</a><a ")
            .replace(/<\/a><br><div class="haku-ver-todos-wrapper">/g, '</a><div class="haku-ver-todos-wrapper">');

        const esFichaProductoUnico =
            totalRegistros === 1 &&
            datos &&
            datos.operacion_consulta === "detalle" &&
            String(texto || "").trim().startsWith("### ");

        if (tieneItemsLista && !mostroBotonVerTodos && !esFichaProductoUnico && datos && datos.url_ver_todos) {
            const sufijoTotal = totalRegistros ? ` (${totalRegistros})` : "";
            htmlFinal += (
                `<div class="haku-ver-todos-wrapper">` +
                `<a href="${escapeHtml(urlVerTodos)}" target="_blank" rel="noopener noreferrer" class="haku-ver-todos-btn">` +
                `${escapeHtml(etiquetaVerTodos)}${sufijoTotal} &rarr;</a>` +
                `</div>`
            );
        }
        return htmlFinal;
    }

    function agregarMensajeChat(remitente, texto, tipo = "normal", datos = null) {
        const chatFeed = document.getElementById("haku-chat-feed");
        if (!chatFeed) return;

        const bubble = document.createElement("div");
        bubble.className = `haku-msg haku-msg-${remitente} haku-msg-type-${tipo}`;

        const hora = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const encabezado = remitente === "usuario" ? "Tú" : "Hacu Inventario";

        let htmlTexto;
        if (remitente === "haku") {
            htmlTexto = formatearMensajeHakuConEnlaces(texto, datos);
        } else {
            htmlTexto = escapeHtml(texto)
                .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                .replace(/`([^`]+)`/g, "<code>$1</code>")
                .replace(/\n/g, "<br>");
        }

        bubble.innerHTML = `
            <div class="msg-header">
                <span class="msg-sender">${encabezado}</span>
                <span class="msg-time">${hora}</span>
            </div>
            <div class="msg-body">${htmlTexto}</div>
        `;

        chatFeed.appendChild(bubble);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function agregarMensajeSistema(texto) {
        const chatFeed = document.getElementById("haku-chat-feed");
        if (!chatFeed) return;

        const info = document.createElement("div");
        info.className = "haku-msg-sistema";
        info.textContent = texto;
        chatFeed.appendChild(info);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function mostrarTarjetaPropuesta(propuesta) {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const badges = {
            ENTRADA: '<span class="badge badge-success">Entrada de stock</span>',
            SALIDA: '<span class="badge badge-danger">Salida de stock</span>',
            PRESTAMO: '<span class="badge badge-warning">Préstamo de artículo</span>',
            DEVOLUCION: '<span class="badge badge-info">Devolución de préstamo</span>',
            TRASLADO_ENTRE: '<span class="badge badge-primary">Traslado entre almacenes</span>',
            CREAR_PRODUCTO: '<span class="badge badge-info">Nuevo producto</span>',
            INHABILITAR_PRODUCTO: '<span class="badge badge-danger">Inhabilitar producto</span>',
            HABILITAR_PRODUCTO: '<span class="badge badge-success">Habilitar producto</span>',
        };

        const badgeHtml = badges[propuesta.tipo_operacion] || '<span class="badge badge-secondary">Operación</span>';
        const esCrearProducto = propuesta.tipo_operacion === "CREAR_PRODUCTO";
        const esDevolucion = propuesta.tipo_operacion === "DEVOLUCION";
        const permiteEditarDestinatario = Boolean(propuesta.destinatario) && !esDevolucion;
        const etiquetaPersona = esDevolucion ? "Responsable del préstamo:" : "Persona / Destinatario:";
        const codigoHtml = propuesta.producto_codigo
            ? ` (<code>${escapeHtml(propuesta.producto_codigo)}</code>)`
            : "";
        const stockInicialVal = (propuesta.cantidad !== undefined && propuesta.cantidad !== null) ? propuesta.cantidad : null;
        const stockMinimoVal = (propuesta.stock_minimo !== undefined && propuesta.stock_minimo !== null)
            ? propuesta.stock_minimo
            : (propuesta.datos_extra && propuesta.datos_extra.stock_minimo !== undefined ? propuesta.datos_extra.stock_minimo : null);
        const ubicacionNombre = propuesta.ubicacion_origen_nombre || (propuesta.datos_extra && propuesta.datos_extra.ubicacion_origen_nombre) || "";
        const ubicacionDestinoNombre = propuesta.ubicacion_destino_nombre || (propuesta.datos_extra && propuesta.datos_extra.ubicacion_destino_nombre) || "";
        const categoriaNombre = propuesta.categoria_nombre || (propuesta.datos_extra && propuesta.datos_extra.categoria_nombre) || "";
        const esPrestableProp = Boolean(
            propuesta.es_prestable !== undefined
                ? propuesta.es_prestable
                : (propuesta.datos_extra && propuesta.datos_extra.es_prestable)
        );
        const esSalida = propuesta.tipo_operacion === "SALIDA";

        contenedor.innerHTML = `
            <div class="propuesta-inner">
                <div class="propuesta-header">
                    <h4>Confirmación de Operación</h4>
                    ${badgeHtml}
                </div>
                <div class="propuesta-detalles">
                    <p><strong>Producto:</strong> ${escapeHtml(propuesta.producto_nombre)}${codigoHtml}</p>
                    ${categoriaNombre ? `<p><strong>Categoría:</strong> ${escapeHtml(categoriaNombre)}</p>` : ""}
                    ${esCrearProducto ? `<p><strong>Tipo de artículo:</strong> ${esPrestableProp ? "Prestable / retornable" : "Consumible (no prestable)"}</p>` : ""}
                    ${esCrearProducto && stockInicialVal !== null ? `<p><strong>Stock inicial:</strong> <span class="cantidad-destacada">${stockInicialVal}</span> unidades</p>` : ""}
                    ${esCrearProducto && stockMinimoVal !== null ? `<p><strong>Stock mínimo:</strong> <span class="cantidad-destacada">${stockMinimoVal}</span> unidades</p>` : ""}
                    ${!esCrearProducto && propuesta.cantidad ? `<p><strong>Cantidad:</strong> <span class="cantidad-destacada">${propuesta.cantidad}</span> unidades</p>` : ""}
                    ${propuesta.almacen_origen_nombre ? `<p><strong>${propuesta.almacen_destino_nombre ? "Almacén origen" : "Almacén"}:</strong> ${escapeHtml(propuesta.almacen_origen_nombre)}</p>` : ""}
                    ${ubicacionNombre ? `<p><strong>Ubicación:</strong> ${escapeHtml(ubicacionNombre)}</p>` : ""}
                    ${propuesta.almacen_destino_nombre ? `<p><strong>Almacén destino:</strong> ${escapeHtml(propuesta.almacen_destino_nombre)}</p>` : ""}
                    ${ubicacionDestinoNombre ? `<p><strong>Ubicación destino:</strong> ${escapeHtml(ubicacionDestinoNombre)}</p>` : ""}
                    ${propuesta.destinatario ? `
                        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:0.5rem 0.75rem; margin:0.5rem 0;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:0.8rem; color:var(--text-muted);">${etiquetaPersona}</span>
                                ${permiteEditarDestinatario ? `<button type="button" id="haku-btn-corregir-dest" style="background:transparent; border:none; color:var(--primary); font-size:0.75rem; cursor:pointer; text-decoration:underline;">✏️ Corregir nombre</button>` : ""}
                            </div>
                            <div style="font-size:1.05rem; font-weight:700; color:var(--accent); margin-top:0.2rem;">
                                👤 ${escapeHtml(propuesta.destinatario)}
                            </div>
                        </div>
                    ` : ""}
                    ${propuesta.motivo ? `<p><strong>Motivo:</strong> ${escapeHtml(propuesta.motivo)}</p>` : ""}
                    ${!esDevolucion && propuesta.observacion ? `<p><strong>Observación:</strong> ${escapeHtml(propuesta.observacion)}</p>` : ""}
                    ${esDevolucion ? `
                        <div style="margin-top:0.55rem;">
                            <label for="haku-dev-observacion" style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">
                                Observación adicional (opcional):
                            </label>
                            <input type="text" id="haku-dev-observacion" class="input-text" style="width:100%;" maxlength="500"
                                   value="${escapeHtml(propuesta.observacion || "")}"
                                   placeholder="Ej. Devuelto en buen estado, accesorios completos...">
                        </div>
                    ` : ""}
                </div>
                <div class="propuesta-acciones">
                    <button type="button" class="btn btn-confirmar" id="haku-btn-confirmar">
                        ✅ Confirmar operación
                    </button>
                    <button type="button" class="btn btn-cancelar" id="haku-btn-cancelar">
                        ❌ Cancelar
                    </button>
                </div>
                <div class="propuesta-nota">
                    También puedes decir <em>"Confirmar"</em> o <em>"Cancelar"</em> por voz.
                </div>
            </div>
        `;

        contenedor.style.display = "block";

        document.getElementById("haku-btn-confirmar")?.addEventListener("click", () => {
            confirmarPropuestaManual(propuesta.id);
        });

        document.getElementById("haku-btn-cancelar")?.addEventListener("click", () => {
            cancelarPropuestaManual(propuesta.id);
        });

        document.getElementById("haku-btn-corregir-dest")?.addEventListener("click", () => {
            mostrarFormularioDestinatario({
                mensaje_pantalla: `Corrige el nombre de la persona para ${esSalida ? "la salida" : "el préstamo"} de ${propuesta.producto_nombre}:`,
            });
        });
    }

    function mostrarFormularioDestinatario(dataAclaracion) {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const candidatos = dataAclaracion.candidatos || [];
        const botonesCandidatos = candidatos.length > 0
            ? `<div style="margin-top:0.5rem; display:flex; flex-wrap:wrap; gap:0.4rem;">
                <span style="font-size:0.75rem; color:var(--text-muted); width:100%;">Candidatos sugeridos:</span>
                ${candidatos.map(c => `<button type="button" class="btn btn-candidato-nom" style="font-size:0.8rem; padding:0.25rem 0.6rem; border:1px solid var(--accent); background:rgba(129,140,248,0.1); border-radius:4px;">👤 ${escapeHtml(c)}</button>`).join("")}
               </div>`
            : "";

        contenedor.innerHTML = `
            <div class="propuesta-inner" style="border-color: var(--warning);">
                <div class="propuesta-header">
                    <h4>Identificación del Destinatario</h4>
                    <span class="badge badge-warning">Entrada Escrita Obligatoria</span>
                </div>
                <div class="propuesta-detalles" style="margin-top:0.5rem;">
                    <p style="color:var(--text-main); font-size:0.9rem;">
                        ${escapeHtml(dataAclaracion.mensaje_pantalla || "Por favor escribe el nombre de la persona para registrar el préstamo:")}
                    </p>
                    ${botonesCandidatos}
                </div>
                <form id="haku-form-destinatario-escrito" style="margin-top:0.75rem; display:flex; gap:0.5rem;">
                    <input type="text" id="haku-input-destinatario-manual" class="input-text" placeholder="Escribe el nombre completo..." required style="flex:1; padding:0.5rem 0.75rem; font-size:0.9rem;" autofocus />
                    <button type="submit" class="btn btn-confirmar" style="padding:0.5rem 1rem;">Guardar</button>
                    <button type="button" id="haku-btn-cancelar-dest" class="btn btn-cancelar" style="padding:0.5rem 0.75rem;">✕</button>
                </form>
            </div>
        `;

        contenedor.style.display = "block";

        const inputNom = document.getElementById("haku-input-destinatario-manual");
        inputNom?.focus();

        // Clic en sugerencias
        contenedor.querySelectorAll(".btn-candidato-nom").forEach(btn => {
            btn.addEventListener("click", () => {
                const nom = btn.textContent.replace("👤", "").trim();
                if (inputNom) inputNom.value = nom;
            });
        });

        // Enviar formulario
        document.getElementById("haku-form-destinatario-escrito")?.addEventListener("submit", async (ev) => {
            ev.preventDefault();
            const val = inputNom ? inputNom.value.trim() : "";
            if (!val) return;

            // Si hay una propuesta activa previa que se está corrigiendo (botón ✏️ Corregir nombre)
            if (state.propuestaActiva && state.propuestaActiva.id) {
                actualizarEstadoVisual("procesando");
                try {
                    const resp = await apiFetch(CONFIG.ENDPOINTS.PROPUESTA_ACTUALIZAR_DESTINATARIO, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            propuesta_id: state.propuestaActiva.id,
                            destinatario: val,
                            conversacion_id: state.conversacionId,
                        }),
                    });
                    const resData = await resp.json();
                    if (resData.exito && resData.propuesta) {
                        state.propuestaActiva = resData.propuesta;
                        mostrarTarjetaPropuesta(resData.propuesta);
                        mostrarToast(`Destinatario actualizado a ${resData.propuesta.destinatario}`, "info");
                    } else {
                        mostrarToast(resData.error || "Error al actualizar destinatario.", "error");
                    }
                } catch (err) {
                    console.error("Error al actualizar destinatario:", err);
                    mostrarToast("Error de conexión al actualizar destinatario.", "error");
                } finally {
                    actualizarEstadoVisual("reposo");
                }
            } else {
                contenedor.style.display = "none";
                agregarMensajeChat("usuario", val);
                enviarComandoHaku(val);
            }
        });

        document.getElementById("haku-btn-cancelar-dest")?.addEventListener("click", () => {
            if (state.propuestaActiva && state.propuestaActiva.id) {
                mostrarTarjetaPropuesta(state.propuestaActiva);
            } else {
                contenedor.style.display = "none";
                enviarComandoHaku("cancelar");
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ESCÁNER DE CÓDIGOS DE BARRAS POR CÁMARA (EAN-13, EAN-8, UPC-A, UPC-E, CODE 128)
    // ─────────────────────────────────────────────────────────────────────────

    let scannerStream = null;
    let scannerAnimationId = null;
    let scannerDetector = null;
    let ultimoCodigoDetectado = null;
    let tiempoUltimaDeteccion = 0;
    let scannerTargetInputId = null;
    let scannerOnDetectadoCb = null;

    const FORMATOS_BARCODE_SOPORTADOS = [
        "ean_13",
        "ean_8",
        "upc_a",
        "upc_e",
        "code_128",
        "code_39",
        "qr_code",
    ];

    async function escanearCodigoBarrasDesdeFoto(file, targetInputId, prefijoForm) {
        if (!file) return;
        const msgDiv = document.getElementById(`${prefijoForm}-barcode-msg`);
        if (msgDiv) {
            msgDiv.style.display = "block";
            msgDiv.style.color = "var(--text-muted)";
            msgDiv.textContent = "🖼️ Analizando foto en busca de código de barras...";
        }

        if (!("BarcodeDetector" in window)) {
            if (msgDiv) {
                msgDiv.style.color = "var(--warning, #f59e0b)";
                msgDiv.textContent = "⚠️ Tu navegador no soporta BarcodeDetector nativo para leer fotos (usa Chrome o Edge).";
            }
            return;
        }

        let detectorFoto = null;
        try {
            detectorFoto = new BarcodeDetector({ formats: FORMATOS_BARCODE_SOPORTADOS });
        } catch (_e1) {
            try {
                detectorFoto = new BarcodeDetector();
            } catch (_e2) {
                detectorFoto = null;
            }
        }
        if (!detectorFoto) {
            if (msgDiv) {
                msgDiv.style.color = "var(--warning, #f59e0b)";
                msgDiv.textContent = "⚠️ No se pudo inicializar el lector de códigos de barras.";
            }
            return;
        }

        try {
            let barcodes = [];
            if (typeof createImageBitmap === "function") {
                const bitmap = await createImageBitmap(file);
                try {
                    barcodes = await detectorFoto.detect(bitmap);
                } finally {
                    if (typeof bitmap.close === "function") bitmap.close();
                }
            } else {
                const imgUrl = URL.createObjectURL(file);
                try {
                    const img = await new Promise((resolve, reject) => {
                        const el = new Image();
                        el.onload = () => resolve(el);
                        el.onerror = reject;
                        el.src = imgUrl;
                    });
                    barcodes = await detectorFoto.detect(img);
                } finally {
                    URL.revokeObjectURL(imgUrl);
                }
            }

            const codigo = (barcodes && barcodes.length > 0 && barcodes[0].rawValue)
                ? barcodes[0].rawValue.trim()
                : "";
            if (!codigo) {
                if (msgDiv) {
                    msgDiv.style.color = "var(--warning, #f59e0b)";
                    msgDiv.textContent = "⚠️ No se detectó ningún código de barras claro en la imagen. Intenta con una foto más nítida.";
                }
                return;
            }

            const inp = document.getElementById(targetInputId);
            if (inp) inp.value = codigo;
            await procesarCodigoBarrasDetectado(codigo, prefijoForm);
        } catch (err) {
            console.error("Error escaneando código de barras desde foto:", err);
            if (msgDiv) {
                msgDiv.style.color = "var(--danger, #ef4444)";
                msgDiv.textContent = "⚠️ Error al procesar la foto seleccionada.";
            }
        }
    }

    async function iniciarEscanerCamara(targetInputId, onDetectado) {
        scannerTargetInputId = targetInputId;
        scannerOnDetectadoCb = onDetectado;

        const container = document.getElementById("haku-scanner-container");
        const video = document.getElementById("haku-scanner-video");
        const statusDiv = document.getElementById("haku-scanner-status");
        const selectCam = document.getElementById("haku-scanner-select-cam");

        if (!container || !video) {
            console.error("Contenedor o elemento de video del escáner no encontrado en el DOM.");
            return;
        }

        // Si ya hay un flujo activo, detenerlo antes de reiniciar
        detenerEscanerCamara();

        container.style.display = "block";
        if (statusDiv) {
            statusDiv.textContent = "Solicitando permisos de cámara...";
            statusDiv.style.color = "var(--text-muted)";
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            if (statusDiv) {
                statusDiv.textContent = "Tu navegador no permite acceso a la cámara. Usa la entrada manual.";
                statusDiv.style.color = "var(--warning, #f59e0b)";
            }
            mostrarToast("Acceso a cámara no disponible en este navegador.", "error");
            return;
        }

        // Verificar soporte del decodificador estándar BarcodeDetector
        if ("BarcodeDetector" in window) {
            try {
                scannerDetector = new BarcodeDetector({ formats: FORMATOS_BARCODE_SOPORTADOS });
            } catch (err) {
                console.warn("No se pudo instanciar BarcodeDetector con todos los formatos, reintentando por defecto:", err);
                try {
                    scannerDetector = new BarcodeDetector();
                } catch (e2) {
                    scannerDetector = null;
                }
            }
        } else {
            scannerDetector = null;
        }

        // Enumerar dispositivos de cámara disponibles
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === "videoinput");
            if (selectCam) {
                selectCam.innerHTML = "";
                videoDevices.forEach((dev, idx) => {
                    const opt = document.createElement("option");
                    opt.value = dev.deviceId;
                    opt.textContent = dev.label || `Cámara ${idx + 1}`;
                    selectCam.appendChild(opt);
                });
                if (videoDevices.length === 0) {
                    const opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "Cámara por defecto";
                    selectCam.appendChild(opt);
                }
            }
        } catch (eDev) {
            console.warn("No se pudieron listar cámaras:", eDev);
        }

        const selectedDeviceId = selectCam ? selectCam.value : null;
        const constraints = {
            video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : { facingMode: "environment" },
            audio: false,
        };

        try {
            scannerStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = scannerStream;
            await video.play();

            if (statusDiv) {
                if (scannerDetector) {
                    statusDiv.textContent = "Apunta la cámara hacia el código de barras (EAN-13, EAN-8, UPC, Code 128)...";
                    statusDiv.style.color = "var(--success, #10b981)";
                } else {
                    statusDiv.textContent = "Cámara activa. La API BarcodeDetector no está activa en tu navegador. Puedes ver el código e ingresarlo manualmente.";
                    statusDiv.style.color = "var(--warning, #f59e0b)";
                }
            }

            // Iniciar bucle de detección si BarcodeDetector está disponible
            if (scannerDetector) {
                iniciarBucleDeteccion(video);
            }
        } catch (err) {
            console.error("Error al iniciar cámara:", err);
            if (statusDiv) {
                if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                    statusDiv.textContent = "Permiso de cámara denegado. Permite el acceso o ingresa el código manualmente.";
                } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                    statusDiv.textContent = "No se encontró ninguna cámara en este dispositivo. Ingresa el código manualmente.";
                } else {
                    statusDiv.textContent = `Error al abrir cámara (${err.name || "desconocido"}). Usa la entrada manual.`;
                }
                statusDiv.style.color = "var(--danger, #ef4444)";
            }
        }
    }

    function iniciarBucleDeteccion(video) {
        let ultimoEscaneo = 0;

        const scanStep = async (timestamp) => {
            if (!scannerStream) return;

            // Procesar cada ~150 ms para no sobrecargar CPU
            if (timestamp - ultimoEscaneo > 150) {
                ultimoEscaneo = timestamp;

                if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && scannerDetector) {
                    try {
                        const barcodes = await scannerDetector.detect(video);
                        if (barcodes && barcodes.length > 0) {
                            const primerCodigo = barcodes[0].rawValue ? barcodes[0].rawValue.trim() : "";
                            const ahora = Date.now();

                            // Evitar captura repetida del mismo código dentro de 3 segundos
                            if (primerCodigo && (primerCodigo !== ultimoCodigoDetectado || ahora - tiempoUltimaDeteccion > 3000)) {
                                ultimoCodigoDetectado = primerCodigo;
                                tiempoUltimaDeteccion = ahora;

                                const statusDiv = document.getElementById("haku-scanner-status");
                                if (statusDiv) {
                                    statusDiv.textContent = `✅ Código detectado: ${primerCodigo}`;
                                    statusDiv.style.color = "var(--success, #10b981)";
                                }

                                // Transferir al input del formulario
                                if (scannerTargetInputId) {
                                    const inp = document.getElementById(scannerTargetInputId);
                                    if (inp) inp.value = primerCodigo;
                                }

                                const cb = scannerOnDetectadoCb;
                                // Detener la cámara tras captura exitosa
                                detenerEscanerCamara();

                                if (typeof cb === "function") {
                                    cb(primerCodigo);
                                }
                                return;
                            }
                        }
                    } catch (scanErr) {
                        // Error de frame individual, continuar bucle
                    }
                }
            }

            scannerAnimationId = requestAnimationFrame(scanStep);
        };

        scannerAnimationId = requestAnimationFrame(scanStep);
    }

    function detenerEscanerCamara() {
        if (scannerAnimationId) {
            cancelAnimationFrame(scannerAnimationId);
            scannerAnimationId = null;
        }

        if (scannerStream) {
            try {
                scannerStream.getTracks().forEach(track => {
                    track.stop();
                });
            } catch (e) {
                console.warn("Error al detener tracks de cámara:", e);
            }
            scannerStream = null;
        }

        const video = document.getElementById("haku-scanner-video");
        if (video) {
            video.srcObject = null;
        }

        const container = document.getElementById("haku-scanner-container");
        if (container) {
            container.style.display = "none";
        }
    }

    async function procesarCodigoBarrasDetectado(codigo, prefijoForm) {
        if (!codigo) return;
        const msgDiv = document.getElementById(`${prefijoForm}-barcode-msg`);
        const inputCod = document.getElementById(`${prefijoForm}-codigo-barras`);
        if (inputCod) inputCod.value = codigo;

        if (msgDiv) {
            msgDiv.style.display = "block";
            msgDiv.style.color = "var(--text-muted)";
            msgDiv.textContent = "🔍 Verificando código de barras en el catálogo...";
        }

        try {
            const url = `${CONFIG.ENDPOINTS.PRODUCTO_CONSULTAR_CODIGO_BARRAS}?codigo=${encodeURIComponent(codigo)}`;
            const resp = await apiFetch(url);
            const data = await resp.json();

            if (!data.valido) {
                if (msgDiv) {
                    msgDiv.textContent = `⚠️ ${data.error || "Código no cumple con el formato estándar EAN/UPC/Code 128."}`;
                    msgDiv.style.color = "var(--danger, #ef4444)";
                }
                return;
            }

            if (data.existe && data.producto) {
                if (msgDiv) {
                    msgDiv.textContent = `⚠️ Este código ya pertenece a '${data.producto.nombre}' (${data.producto.codigo_interno}).`;
                    msgDiv.style.color = "var(--danger, #ef4444)";
                }
                return;
            }

            if (data.encontrado_en_catalogo && data.datos_sugeridos) {
                const sug = data.datos_sugeridos;
                if (msgDiv) {
                    msgDiv.textContent = `✅ Código ${data.formato || ""} válido. Datos sugeridos cargados desde catálogo.`;
                    msgDiv.style.color = "var(--success, #10b981)";
                }
                const inputNom = document.getElementById(`${prefijoForm}-nombre`);
                const inputEmp = document.getElementById(`${prefijoForm}-nombre-empaque`);
                const inputUds = document.getElementById(`${prefijoForm}-uds-empaque`);
                const inputDesc = document.getElementById(`${prefijoForm}-descripcion`);

                if (inputNom && !inputNom.value && sug.nombre) inputNom.value = sug.nombre;
                if (inputEmp && !inputEmp.value && sug.nombre_empaque) inputEmp.value = sug.nombre_empaque;
                if (inputUds && !inputUds.value && sug.unidades_por_empaque) inputUds.value = sug.unidades_por_empaque;
                if (inputDesc && !inputDesc.value && sug.descripcion) inputDesc.value = sug.descripcion;
            } else {
                if (msgDiv) {
                    msgDiv.textContent = `✅ Código ${data.formato || ""} válido y disponible para registrar.`;
                    msgDiv.style.color = "var(--success, #10b981)";
                }
            }
        } catch (err) {
            console.error("Error al consultar código de barras:", err);
            if (msgDiv) {
                msgDiv.textContent = "Error de conexión al verificar el código de barras.";
                msgDiv.style.color = "var(--warning, #f59e0b)";
            }
        }
    }

    function obtenerDatosCatalogoCrearProducto() {
        const dataEl = document.getElementById("haku-crear-producto-datos");
        if (!dataEl) return { categorias: [], almacenes: [], ubicaciones: [] };
        try {
            const parsed = JSON.parse(dataEl.textContent || "{}");
            return {
                categorias: parsed.categorias || [],
                almacenes: parsed.almacenes || [],
                ubicaciones: parsed.ubicaciones || [],
            };
        } catch (_err) {
            return { categorias: [], almacenes: [], ubicaciones: [] };
        }
    }

    function mostrarFormularioCrearProducto(datos, prefill = {}) {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const baseCatalogo = obtenerDatosCatalogoCrearProducto();
        const categoriasAll = (datos && datos.categorias && datos.categorias.length) ? datos.categorias : baseCatalogo.categorias;
        const almacenesAll = (datos && datos.almacenes && datos.almacenes.length) ? datos.almacenes : baseCatalogo.almacenes;
        const ubicacionesAll = (datos && datos.ubicaciones && datos.ubicaciones.length) ? datos.ubicaciones : baseCatalogo.ubicaciones;

        const nombreInit = prefill.nombre ? String(prefill.nombre) : "";
        const almInit = prefill.almacen_id ? String(prefill.almacen_id) : "";
        const catInit = prefill.categoria_id ? String(prefill.categoria_id) : "";
        const ubiInit = prefill.ubicacion_id ? String(prefill.ubicacion_id) : "";
        const stockIniInit = (prefill.stock_inicial !== undefined && prefill.stock_inicial !== null) ? String(prefill.stock_inicial) : "0";
        const stockMinInit = (prefill.stock_minimo !== undefined && prefill.stock_minimo !== null) ? String(prefill.stock_minimo) : "";
        const esPrestableInit = Boolean(prefill.es_prestable);

        const almacenesHtml = `<option value="">-- Selecciona un almacén --</option>` + almacenesAll.map(
            a => `<option value="${a.id}" ${String(a.id) === almInit ? "selected" : ""}>${escapeHtml(a.nombre)}</option>`
        ).join("");

        contenedor.innerHTML = `
            <div class="propuesta-inner">
                <div class="propuesta-header">
                    <h4>Registro de Nuevo Producto</h4>
                    <span class="badge badge-info">Formulario</span>
                </div>

                <form id="haku-form-crear-producto" class="haku-form-interactivo">
                    <div class="haku-form-field">
                        <label class="haku-form-label" for="haku-crear-nombre">Nombre del producto <span class="haku-form-required">*</span></label>
                        <input type="text" id="haku-crear-nombre" required class="input-text" value="${escapeHtml(nombreInit)}" placeholder="Ej. Sensor Ultrasónico HC-SR04">
                    </div>

                    <div class="haku-form-grid-2">
                        <div class="haku-form-field">
                            <label class="haku-form-label" for="haku-crear-almacen">Almacén</label>
                            <select id="haku-crear-almacen" class="input-text haku-form-select">
                                ${almacenesHtml}
                            </select>
                        </div>
                        <div class="haku-form-field">
                            <label class="haku-form-label" for="haku-crear-categoria">Categoría <span class="haku-form-required">*</span></label>
                            <select id="haku-crear-categoria" required class="input-text haku-form-select"></select>
                        </div>
                    </div>

                    <div class="haku-form-grid-2">
                        <div class="haku-form-field">
                            <label class="haku-form-label" for="haku-crear-ubicacion">Ubicación física</label>
                            <select id="haku-crear-ubicacion" class="input-text haku-form-select"></select>
                        </div>
                        <div class="haku-form-grid-2">
                            <div class="haku-form-field">
                                <label class="haku-form-label" for="haku-crear-stock-inicial">Stock inicial</label>
                                <input type="number" id="haku-crear-stock-inicial" value="${escapeHtml(stockIniInit)}" min="0" class="input-text">
                            </div>
                            <div class="haku-form-field">
                                <label class="haku-form-label" for="haku-crear-stock-minimo">Stock mínimo <span class="haku-form-required">*</span></label>
                                <input type="number" id="haku-crear-stock-minimo" min="0" required class="input-text" value="${escapeHtml(stockMinInit)}" placeholder="0 o más">
                            </div>
                        </div>
                    </div>

                    <div class="haku-form-field" style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:0.55rem 0.75rem;">
                        <label for="haku-crear-es-prestable" style="display:flex; align-items:flex-start; gap:0.55rem; cursor:pointer; margin:0;">
                            <input type="checkbox" id="haku-crear-es-prestable" ${esPrestableInit ? "checked" : ""} style="margin-top:0.2rem; width:16px; height:16px; accent-color:var(--accent); cursor:pointer;">
                            <span>
                                <strong style="font-size:0.84rem; color:var(--text-main); display:block;">¿Es prestable / retornable?</strong>
                                <span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:2px;">Actívalo para herramientas o equipos que se prestan y devuelven. Déjalo desmarcado para consumibles.</span>
                            </span>
                        </label>
                    </div>

                    <div class="haku-form-field" style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:0.6rem;">
                        <label class="haku-form-label" for="haku-crear-codigo-barras">Código de barras (opcional)</label>
                        <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                            <input type="text" id="haku-crear-codigo-barras" class="input-text" style="flex:1; min-width:160px;" placeholder="EAN / UPC / Code 128">
                            <button type="button" id="haku-btn-escanear-camara-crear" class="btn btn-primary" style="white-space:nowrap; font-size:0.8rem;">
                                📷 Escanear
                            </button>
                            <button type="button" id="haku-btn-foto-codigo-crear" class="btn" style="white-space:nowrap; font-size:0.8rem;">
                                🖼️ Foto
                            </button>
                            <button type="button" id="haku-btn-verificar-codigo-crear" class="btn" style="white-space:nowrap; font-size:0.8rem;">
                                🔍 Verificar
                            </button>
                            <input type="file" id="haku-input-foto-codigo-crear" accept="image/*" capture="environment" style="display:none;">
                        </div>
                        <div id="haku-crear-barcode-msg" style="display:none; font-size:0.8rem; margin-top:0.4rem;"></div>
                        <div id="haku-scanner-container" style="display:none; background:rgba(0,0,0,0.85); border:1px solid var(--accent); border-radius:8px; padding:0.75rem; margin-top:0.6rem; text-align:center;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                                <span style="font-size:0.85rem; font-weight:600; color:var(--text-main);">📸 Escáner de Códigos de Barras</span>
                                <button type="button" id="haku-scanner-btn-cerrar" class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;">✕ Cerrar visor</button>
                            </div>
                            <video id="haku-scanner-video" style="width:100%; max-height:220px; border-radius:4px; object-fit:cover; background:#000;" playsinline autoplay muted></video>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem; gap:0.5rem;">
                                <select id="haku-scanner-select-cam" class="input-text" style="font-size:0.75rem; flex:1;">
                                    <option value="">Cámara trasera / por defecto</option>
                                </select>
                                <button type="button" id="haku-scanner-btn-cambiar-cam" class="btn" style="font-size:0.75rem;">🔄 Cambiar</button>
                            </div>
                            <div id="haku-scanner-status" style="font-size:0.8rem; color:var(--text-muted); margin-top:0.4rem;">Iniciando...</div>
                        </div>
                    </div>

                    <div class="haku-form-grid-2">
                        <div class="haku-form-field">
                            <label class="haku-form-label" for="haku-crear-nombre-empaque">Empaque (opcional)</label>
                            <input type="text" id="haku-crear-nombre-empaque" class="input-text" placeholder="Unidad / Caja / Paquete">
                        </div>
                        <div class="haku-form-field">
                            <label class="haku-form-label" for="haku-crear-uds-empaque">Uds por empaque</label>
                            <input type="number" id="haku-crear-uds-empaque" min="1" class="input-text" placeholder="Ej. 12">
                        </div>
                    </div>

                    <div class="haku-form-field">
                        <label class="haku-form-label" for="haku-crear-descripcion">Descripción</label>
                        <textarea id="haku-crear-descripcion" class="input-text haku-form-textarea" placeholder="Detalles técnicos o notas"></textarea>
                    </div>

                    <div class="propuesta-acciones">
                        <button type="submit" class="btn btn-confirmar">Guardar producto</button>
                        <button type="button" class="btn btn-cancelar" id="haku-btn-cancelar-crear">Cancelar</button>
                    </div>
                </form>
            </div>
        `;
        contenedor.style.display = "block";

        const selAlm = document.getElementById("haku-crear-almacen");
        const selCat = document.getElementById("haku-crear-categoria");
        const selUbi = document.getElementById("haku-crear-ubicacion");
        const almacenesPorId = {};
        almacenesAll.forEach(a => { almacenesPorId[String(a.id)] = a.nombre; });

        function actualizarDependientesAlmacen(selectedCatId = "", selectedUbiId = "") {
            const almIdNum = selAlm && selAlm.value ? parseInt(selAlm.value, 10) : null;
            const catsFiltradas = almIdNum
                ? categoriasAll.filter(c => !c.almacen_principal_id || c.almacen_principal_id === almIdNum)
                : categoriasAll;
            if (selCat) {
                selCat.innerHTML = `<option value="">-- Selecciona una categoría --</option>` + catsFiltradas.map(
                    c => `<option value="${c.id}" ${String(c.id) === String(selectedCatId) ? "selected" : ""}>${escapeHtml(c.nombre)} (${escapeHtml(c.prefijo_codigo)})</option>`
                ).join("");
            }
            const ubisFiltradas = almIdNum
                ? ubicacionesAll.filter(u => u.almacen_id === almIdNum)
                : ubicacionesAll;
            if (selUbi) {
                selUbi.innerHTML = `<option value="">-- Sin ubicación específica --</option>` + ubisFiltradas.map(u => {
                    const almSufijo = (!almIdNum && u.almacen_id && almacenesPorId[String(u.almacen_id)])
                        ? ` (${almacenesPorId[String(u.almacen_id)]})`
                        : "";
                    return `<option value="${u.id}" ${String(u.id) === String(selectedUbiId) ? "selected" : ""}>${escapeHtml(u.nombre + almSufijo)}</option>`;
                }).join("");
            }
        }

        actualizarDependientesAlmacen(catInit, ubiInit);

        selAlm?.addEventListener("change", () => {
            const almIdNum = selAlm.value ? parseInt(selAlm.value, 10) : null;
            const catActual = selCat?.value ? categoriasAll.find(c => String(c.id) === String(selCat.value)) : null;
            const ubiActual = selUbi?.value ? ubicacionesAll.find(u => String(u.id) === String(selUbi.value)) : null;
            const catConservada = (catActual && (!almIdNum || !catActual.almacen_principal_id || catActual.almacen_principal_id === almIdNum))
                ? String(catActual.id) : "";
            const ubiConservada = (ubiActual && (!almIdNum || ubiActual.almacen_id === almIdNum))
                ? String(ubiActual.id) : "";
            actualizarDependientesAlmacen(catConservada, ubiConservada);
        });

        selUbi?.addEventListener("change", () => {
            const ubiSel = selUbi.value ? ubicacionesAll.find(u => String(u.id) === String(selUbi.value)) : null;
            if (ubiSel && ubiSel.almacen_id && selAlm && String(selAlm.value) !== String(ubiSel.almacen_id)) {
                selAlm.value = String(ubiSel.almacen_id);
                const catActual = selCat?.value ? categoriasAll.find(c => String(c.id) === String(selCat.value)) : null;
                const catConservada = (catActual && (!catActual.almacen_principal_id || catActual.almacen_principal_id === ubiSel.almacen_id))
                    ? String(catActual.id) : "";
                actualizarDependientesAlmacen(catConservada, String(ubiSel.id));
            }
        });

        selCat?.addEventListener("change", () => {
            const catSel = selCat.value ? categoriasAll.find(c => String(c.id) === String(selCat.value)) : null;
            if (catSel && catSel.almacen_principal_id && selAlm && String(selAlm.value) !== String(catSel.almacen_principal_id)) {
                selAlm.value = String(catSel.almacen_principal_id);
                const ubiActual = selUbi?.value ? ubicacionesAll.find(u => String(u.id) === String(selUbi.value)) : null;
                const ubiConservada = (ubiActual && ubiActual.almacen_id === catSel.almacen_principal_id)
                    ? String(ubiActual.id) : "";
                actualizarDependientesAlmacen(String(catSel.id), ubiConservada);
            }
        });

        document.getElementById("haku-btn-escanear-camara-crear")?.addEventListener("click", () => {
            iniciarEscanerCamara("haku-crear-codigo-barras", (cod) => {
                procesarCodigoBarrasDetectado(cod, "haku-crear");
            });
        });

        document.getElementById("haku-btn-foto-codigo-crear")?.addEventListener("click", () => {
            const inpFoto = document.getElementById("haku-input-foto-codigo-crear");
            if (inpFoto) {
                inpFoto.value = "";
                inpFoto.click();
            }
        });

        document.getElementById("haku-input-foto-codigo-crear")?.addEventListener("change", (ev) => {
            const file = ev.target && ev.target.files ? ev.target.files[0] : null;
            if (file) escanearCodigoBarrasDesdeFoto(file, "haku-crear-codigo-barras", "haku-crear");
        });

        document.getElementById("haku-btn-verificar-codigo-crear")?.addEventListener("click", () => {
            const cod = document.getElementById("haku-crear-codigo-barras")?.value?.trim();
            if (cod) procesarCodigoBarrasDetectado(cod, "haku-crear");
        });

        document.getElementById("haku-scanner-btn-cerrar")?.addEventListener("click", detenerEscanerCamara);
        document.getElementById("haku-scanner-btn-cambiar-cam")?.addEventListener("click", () => {
            const selCam = document.getElementById("haku-scanner-select-cam");
            if (selCam && selCam.options.length > 1) {
                selCam.selectedIndex = (selCam.selectedIndex + 1) % selCam.options.length;
                iniciarEscanerCamara("haku-crear-codigo-barras", (cod) => {
                    procesarCodigoBarrasDetectado(cod, "haku-crear");
                });
            }
        });

        document.getElementById("haku-btn-cancelar-crear")?.addEventListener("click", () => {
            detenerEscanerCamara();
            ocultarTarjetaPropuesta();
            if (prefill && prefill._conversacional) {
                enviarComandoHaku("Cancelar");
            } else {
                agregarMensajeChat("haku", "Operación de registro cancelada por el usuario.");
            }
        });

        document.getElementById("haku-form-crear-producto")?.addEventListener("submit", async (e) => {
            e.preventDefault();
            const almVal = document.getElementById("haku-crear-almacen")?.value;
            const ubiVal = document.getElementById("haku-crear-ubicacion")?.value;
            const stockMinVal = document.getElementById("haku-crear-stock-minimo")?.value;
            const payload = {
                nombre: document.getElementById("haku-crear-nombre").value.trim(),
                categoria_id: parseInt(document.getElementById("haku-crear-categoria").value, 10),
                almacen_id: almVal ? parseInt(almVal, 10) : null,
                ubicacion_id: ubiVal ? parseInt(ubiVal, 10) : null,
                stock_inicial: parseInt(document.getElementById("haku-crear-stock-inicial")?.value, 10) || 0,
                codigo_barras: document.getElementById("haku-crear-codigo-barras")?.value?.trim() || null,
                stock_minimo: (stockMinVal !== undefined && stockMinVal !== "") ? parseInt(stockMinVal, 10) : null,
                es_prestable: Boolean(document.getElementById("haku-crear-es-prestable")?.checked),
                nombre_empaque: document.getElementById("haku-crear-nombre-empaque").value.trim(),
                unidades_por_empaque: document.getElementById("haku-crear-uds-empaque").value ? parseInt(document.getElementById("haku-crear-uds-empaque").value, 10) : null,
                descripcion: document.getElementById("haku-crear-descripcion").value.trim(),
            };
            actualizarEstadoVisual("procesando");
            try {
                const resp = await apiFetch(CONFIG.ENDPOINTS.PRODUCTO_CREAR, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const resData = await resp.json();
                if (resData.error) {
                    mostrarToast(resData.error, "error");
                    agregarMensajeChat("haku", `⚠️ No se pudo crear: ${resData.error}`);
                } else {
                    ocultarTarjetaPropuesta();
                    mostrarToast("Producto registrado exitosamente.", "info");
                    agregarMensajeChat("haku", `✅ ${resData.mensaje_pantalla}`);
                    if (resData.mensaje_verbal) reproducirVozCliente(resData.mensaje_verbal);
                }
            } catch (err) {
                console.error("Error creando producto:", err);
                mostrarToast("Error de conexión al registrar producto.", "error");
            } finally {
                actualizarEstadoVisual("reposo");
            }
        });
    }

    function mostrarFormularioEditarProducto(datos) {
        detenerEscanerCamara();
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const prod = datos.producto || {};
        const categorias = datos.categorias || [];
        const categoriasHtml = categorias.map(c => `
            <option value="${c.id}" ${c.id === prod.categoria_id ? "selected" : ""}>
                ${escapeHtml(c.nombre)}${c.prefijo_codigo ? ` (${escapeHtml(c.prefijo_codigo)})` : ""}
            </option>
        `).join("");

        contenedor.innerHTML = `
            <div class="propuesta-inner" style="border-color: var(--accent);">
                <div class="propuesta-header">
                    <h4>Edición de Producto: ${escapeHtml(prod.nombre)}</h4>
                    <span class="badge badge-warning"><code>${escapeHtml(prod.codigo_interno)}</code></span>
                </div>

                <!-- Visor integrado del escáner de cámara para edición -->
                <div id="haku-scanner-container" style="display:none; background:rgba(0,0,0,0.85); border:1px solid var(--accent); border-radius:8px; padding:0.75rem; margin-top:0.75rem; text-align:center;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                        <span style="font-size:0.85rem; font-weight:600; color:var(--text-main);">📸 Escáner de Códigos de Barras</span>
                        <button type="button" id="haku-scanner-btn-cerrar" class="btn" style="padding:0.2rem 0.5rem; font-size:0.75rem;">✕ Cerrar visor</button>
                    </div>
                    <video id="haku-scanner-video" style="width:100%; max-height:220px; border-radius:4px; object-fit:cover; background:#000;" playsinline autoplay muted></video>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem; gap:0.5rem;">
                        <select id="haku-scanner-select-cam" class="input-text" style="font-size:0.75rem; flex:1;">
                            <option value="">Cámara trasera / por defecto</option>
                        </select>
                        <button type="button" id="haku-scanner-btn-cambiar-cam" class="btn" style="font-size:0.75rem;">🔄 Cambiar</button>
                    </div>
                    <div id="haku-scanner-status" style="font-size:0.8rem; color:var(--text-muted); margin-top:0.4rem;">Iniciando...</div>
                </div>

                <form id="haku-form-editar-producto" class="haku-form-interactivo" style="margin-top: 1rem; display: flex; flex-direction: column; gap: 0.75rem;">
                    <input type="hidden" id="haku-editar-id" value="${prod.id}">
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nombre del producto *</label>
                        <input type="text" id="haku-editar-nombre" required class="input-text" style="width:100%;" value="${escapeHtml(prod.nombre || '')}">
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Categoría *</label>
                        <select id="haku-editar-categoria" required class="input-text" style="width:100%; background:var(--bg-main); color:var(--text-main);">
                            ${categoriasHtml}
                        </select>
                    </div>

                    <!-- Código de barras con botón de escaneo con cámara y por foto -->
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:0.6rem;">
                        <label style="display:block; font-size:0.8rem; font-weight:600; color:var(--text-main); margin-bottom:0.25rem;">Código de barras (opcional)</label>
                        <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                            <input type="text" id="haku-editar-codigo-barras" class="input-text" style="flex:1; min-width:160px;" value="${escapeHtml(prod.codigo_barras || '')}" placeholder="EAN / UPC">
                            <button type="button" id="haku-btn-escanear-camara-edit" class="btn btn-primary" style="white-space:nowrap; font-size:0.8rem;">
                                📷 Escanear
                            </button>
                            <button type="button" id="haku-btn-foto-codigo-edit" class="btn" style="white-space:nowrap; font-size:0.8rem;">
                                🖼️ Foto
                            </button>
                            <button type="button" id="haku-btn-verificar-codigo-edit" class="btn" style="white-space:nowrap; font-size:0.8rem;">
                                🔍 Verificar
                            </button>
                            <input type="file" id="haku-input-foto-codigo-edit" accept="image/*" capture="environment" style="display:none;">
                        </div>
                        <div id="haku-editar-barcode-msg" style="display:none; font-size:0.8rem; margin-top:0.4rem;"></div>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Stock mínimo</label>
                            <input type="number" id="haku-editar-stock-minimo" min="0" class="input-text" style="width:100%;" value="${prod.stock_minimo !== undefined ? prod.stock_minimo : 0}">
                        </div>
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Empaque (ej. Caja, Paquete)</label>
                            <input type="text" id="haku-editar-nombre-empaque" class="input-text" style="width:100%;" value="${escapeHtml(prod.nombre_empaque || '')}">
                        </div>
                    </div>
                    <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:0.55rem 0.75rem;">
                        <label for="haku-editar-es-prestable" style="display:flex; align-items:flex-start; gap:0.55rem; cursor:pointer; margin:0;">
                            <input type="checkbox" id="haku-editar-es-prestable" ${prod.es_prestable ? "checked" : ""} style="margin-top:0.2rem; width:16px; height:16px; accent-color:var(--accent); cursor:pointer;">
                            <span>
                                <strong style="font-size:0.84rem; color:var(--text-main); display:block;">¿Es prestable / retornable?</strong>
                                <span style="font-size:0.75rem; color:var(--text-muted); display:block; margin-top:2px;">Actívalo para herramientas o equipos que se prestan y devuelven. Déjalo desmarcado para consumibles.</span>
                            </span>
                        </label>
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Uds por empaque</label>
                        <input type="number" id="haku-editar-uds-empaque" min="1" class="input-text" style="width:100%;" value="${prod.unidades_por_empaque || ''}">
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Descripción</label>
                        <textarea id="haku-editar-descripcion" class="input-text" style="width:100%; height:60px;">${escapeHtml(prod.descripcion || '')}</textarea>
                    </div>
                    <div class="propuesta-acciones" style="margin-top: 0.5rem;">
                        <button type="submit" class="btn btn-confirmar">💾 Guardar cambios</button>
                        <button type="button" class="btn btn-cancelar" id="haku-btn-cancelar-editar">❌ Cancelar</button>
                    </div>
                </form>
            </div>
        `;
        contenedor.style.display = "block";

        document.getElementById("haku-btn-escanear-camara-edit")?.addEventListener("click", () => {
            iniciarEscanerCamara("haku-editar-codigo-barras", (cod) => {
                procesarCodigoBarrasDetectado(cod, "haku-editar");
            });
        });

        document.getElementById("haku-btn-foto-codigo-edit")?.addEventListener("click", () => {
            const inpFoto = document.getElementById("haku-input-foto-codigo-edit");
            if (inpFoto) {
                inpFoto.value = "";
                inpFoto.click();
            }
        });

        document.getElementById("haku-input-foto-codigo-edit")?.addEventListener("change", (ev) => {
            const file = ev.target && ev.target.files ? ev.target.files[0] : null;
            if (file) escanearCodigoBarrasDesdeFoto(file, "haku-editar-codigo-barras", "haku-editar");
        });

        document.getElementById("haku-btn-verificar-codigo-edit")?.addEventListener("click", () => {
            const val = document.getElementById("haku-editar-codigo-barras").value.trim();
            if (val) procesarCodigoBarrasDetectado(val, "haku-editar");
        });

        document.getElementById("haku-btn-cancelar-editar")?.addEventListener("click", () => {
            detenerEscanerCamara();
            ocultarTarjetaPropuesta();
            agregarMensajeChat("haku", "Edición cancelada por el usuario.");
        });

        document.getElementById("haku-form-editar-producto")?.addEventListener("submit", async (e) => {
            e.preventDefault();
            detenerEscanerCamara();
            const stockMinVal = document.getElementById("haku-editar-stock-minimo")?.value;
            const payload = {
                producto_id: parseInt(document.getElementById("haku-editar-id").value, 10),
                nombre: document.getElementById("haku-editar-nombre").value.trim(),
                categoria_id: parseInt(document.getElementById("haku-editar-categoria").value, 10),
                codigo_barras: document.getElementById("haku-editar-codigo-barras").value.trim() || null,
                stock_minimo: (stockMinVal !== undefined && stockMinVal !== "") ? parseInt(stockMinVal, 10) : null,
                es_prestable: Boolean(document.getElementById("haku-editar-es-prestable")?.checked),
                nombre_empaque: document.getElementById("haku-editar-nombre-empaque").value.trim(),
                unidades_por_empaque: document.getElementById("haku-editar-uds-empaque").value ? parseInt(document.getElementById("haku-editar-uds-empaque").value, 10) : null,
                descripcion: document.getElementById("haku-editar-descripcion").value.trim(),
            };
            actualizarEstadoVisual("procesando");
            try {
                const resp = await apiFetch(CONFIG.ENDPOINTS.PRODUCTO_EDITAR, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const resData = await resp.json();
                if (resData.error) {
                    mostrarToast(resData.error, "error");
                    agregarMensajeChat("haku", `⚠️ No se pudo actualizar: ${resData.error}`);
                } else {
                    ocultarTarjetaPropuesta();
                    mostrarToast("Producto actualizado con éxito.", "info");
                    agregarMensajeChat("haku", `✅ ${resData.mensaje_pantalla}`);
                    if (resData.mensaje_verbal) reproducirVozCliente(resData.mensaje_verbal);
                }
            } catch (err) {
                console.error("Error actualizando producto:", err);
                mostrarToast("Error de conexión al actualizar producto.", "error");
            } finally {
                actualizarEstadoVisual("reposo");
            }
        });
    }

    function mostrarFormularioCrearUsuario(datos) {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const roles = datos.roles || [];
        const rolesHtml = roles.map(r => `<option value="${escapeHtml(r.valor)}">${escapeHtml(r.nombre)}</option>`).join("");
        const almacenes = datos.almacenes || [];
        const almacenesCheckboxes = almacenes.map(a => `
            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
                <input type="checkbox" name="haku-crear-almacenes-multi" value="${a.id}" class="haku-chk-alm">
                ${escapeHtml(a.nombre)}
            </label>
        `).join("");

        contenedor.innerHTML = `
            <div class="propuesta-inner" style="border-color: var(--primary);">
                <div class="propuesta-header">
                    <h4>Registro de Nuevo Usuario</h4>
                    <span class="badge badge-info">Gestión de Usuarios</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                    Rol de operador: <strong>${escapeHtml(datos.usuario_actor_rol)}</strong>
                </div>
                <form id="haku-form-crear-usuario" class="haku-form-interactivo" style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nombre de usuario (login) *</label>
                            <input type="text" id="haku-user-crear-username" required class="input-text" style="width:100%;" placeholder="ej. jpolo">
                        </div>
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nombre completo *</label>
                            <input type="text" id="haku-user-crear-nombre" required class="input-text" style="width:100%;" placeholder="ej. Juan Polo">
                        </div>
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Rol asignado *</label>
                        <select id="haku-user-crear-rol" required class="input-text" style="width:100%; background:var(--bg-main); color:var(--text-main);">
                            ${rolesHtml}
                        </select>
                    </div>
                    <div id="haku-user-crear-almacenes-bloque" style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:0.75rem;">
                        <label style="display:block; font-size:0.8rem; font-weight:600; color:var(--text-main); margin-bottom:0.4rem;">
                            Asignación de Almacenes <span id="haku-user-almacen-req" style="color:var(--danger, #ff4444);">*</span>
                        </label>
                        <div style="margin-bottom:0.5rem;">
                            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; font-weight:500; cursor:pointer;">
                                <input type="checkbox" id="haku-user-crear-todos-almacenes">
                                <strong>Acceso a todos los almacenes</strong>
                            </label>
                        </div>
                        <div id="haku-user-crear-lista-almacenes" style="display:flex; flex-direction:column; gap:0.35rem; max-height:140px; overflow-y:auto; padding-left:0.25rem;">
                            ${almacenesCheckboxes}
                        </div>
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Contraseña inicial *</label>
                        <input type="password" id="haku-user-crear-password" required class="input-text" style="width:100%;" placeholder="Mínimo 8 caracteres, alfanumérica">
                        <div id="haku-user-crear-password-error" style="display:none; color:var(--danger, #ef4444); font-size:0.8rem; margin-top:0.35rem; font-weight:500;"></div>
                    </div>
                    <div class="propuesta-acciones" style="margin-top: 0.5rem;">
                        <button type="submit" class="btn btn-confirmar">✅ Crear usuario</button>
                        <button type="button" class="btn btn-cancelar" id="haku-btn-cancelar-crear-usuario">❌ Cancelar</button>
                    </div>
                </form>
            </div>
        `;
        contenedor.style.display = "block";

        const selRol = document.getElementById("haku-user-crear-rol");
        const bloqueAlm = document.getElementById("haku-user-crear-almacenes-bloque");
        const chkTodos = document.getElementById("haku-user-crear-todos-almacenes");
        const listaAlm = document.getElementById("haku-user-crear-lista-almacenes");

        if (chkTodos && listaAlm) {
            chkTodos.addEventListener("change", () => {
                const checkboxes = listaAlm.querySelectorAll(".haku-chk-alm");
                checkboxes.forEach(cb => {
                    cb.disabled = chkTodos.checked;
                    if (chkTodos.checked) cb.checked = false;
                });
            });
            listaAlm.querySelectorAll(".haku-chk-alm").forEach(cb => {
                cb.addEventListener("change", () => {
                    if (cb.checked && chkTodos.checked) {
                        chkTodos.checked = false;
                    }
                });
            });
        }

        if (selRol && bloqueAlm) {
            selRol.addEventListener("change", () => {
                if (selRol.value === "USUARIO") {
                    bloqueAlm.style.display = "block";
                } else {
                    bloqueAlm.style.display = "none";
                }
            });
            selRol.dispatchEvent(new Event("change"));
        }

        document.getElementById("haku-btn-cancelar-crear-usuario")?.addEventListener("click", () => {
            ocultarTarjetaPropuesta();
            agregarMensajeChat("haku", "Creación de usuario cancelada.");
        });

        document.getElementById("haku-form-crear-usuario")?.addEventListener("submit", async (e) => {
            e.preventDefault();
            const rolVal = document.getElementById("haku-user-crear-rol").value.trim();
            const accesoTodos = rolVal === "USUARIO" && document.getElementById("haku-user-crear-todos-almacenes").checked;
            const almsSeleccionados = [];
            document.querySelectorAll("#haku-user-crear-lista-almacenes .haku-chk-alm:checked").forEach(cb => {
                almsSeleccionados.push(parseInt(cb.value, 10));
            });

            if (rolVal === "USUARIO" && !accesoTodos && almsSeleccionados.length === 0) {
                mostrarToast("Debes seleccionar al menos un almacén o marcar acceso a todos.", "error");
                return;
            }

            const payload = {
                username: document.getElementById("haku-user-crear-username").value.trim(),
                nombre_completo: document.getElementById("haku-user-crear-nombre").value.trim(),
                rol: rolVal,
                almacen_asignado_id: almsSeleccionados.length > 0 ? almsSeleccionados[0] : null,
                almacenes_asignados_ids: almsSeleccionados,
                acceso_todos_almacenes: accesoTodos,
                password: document.getElementById("haku-user-crear-password").value,
            };

            actualizarEstadoVisual("procesando");
            try {
                const resp = await apiFetch(CONFIG.ENDPOINTS.USUARIO_CREAR, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const resData = await resp.json();
                if (resData.error) {
                    mostrarToast(resData.error, "error");
                    let detalleTexto = "";
                    if (resData.detalles) {
                        detalleTexto = Object.values(resData.detalles).join(". ");
                    }
                    const errorMsg = detalleTexto ? `${resData.error}: ${detalleTexto}` : resData.error;

                    const errorDiv = document.getElementById("haku-user-crear-password-error");
                    const passInput = document.getElementById("haku-user-crear-password");
                    if (errorDiv) {
                        errorDiv.textContent = `⚠️ ${errorMsg}`;
                        errorDiv.style.display = "block";
                    }
                    if (passInput) {
                        passInput.value = "";
                        passInput.focus();
                        passInput.style.borderColor = "var(--danger, #ef4444)";
                    }
                    // Mantener el formulario abierto y NO enviar mensajes de error de contraseña al chat
                } else {
                    ocultarTarjetaPropuesta();
                    mostrarToast("Usuario creado con éxito.", "info");
                    agregarMensajeChat("haku", `✅ ${resData.mensaje_pantalla}`);
                    if (resData.mensaje_verbal) reproducirVozCliente(resData.mensaje_verbal);
                }
            } catch (err) {
                console.error("Error al crear usuario:", err);
                mostrarToast("Error de conexión al crear usuario.", "error");
            } finally {
                actualizarEstadoVisual("reposo");
            }
        });
    }

    function mostrarFormularioEditarUsuario(datos) {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (!contenedor) return;

        const u = datos.usuario || {};
        const roles = datos.roles || [];
        const rolesHtml = roles.map(r => `<option value="${escapeHtml(r.valor)}" ${u.rol === r.valor ? "selected" : ""}>${escapeHtml(r.nombre)}</option>`).join("");
        const almacenes = datos.almacenes || [];
        const assignedIds = new Set(u.almacenes_asignados_ids || (u.almacen_asignado_id ? [u.almacen_asignado_id] : []));
        const almacenesCheckboxes = almacenes.map(a => `
            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; cursor:pointer;">
                <input type="checkbox" name="haku-editar-almacenes-multi" value="${a.id}" class="haku-chk-alm-edit" ${assignedIds.has(a.id) ? "checked" : ""}>
                ${escapeHtml(a.nombre)}
            </label>
        `).join("");

        contenedor.innerHTML = `
            <div class="propuesta-inner" style="border-color: var(--primary);">
                <div class="propuesta-header">
                    <h4>Editar Usuario: ${escapeHtml(u.username || "")}</h4>
                    <span class="badge badge-warning">Edición de Cuenta</span>
                </div>
                <form id="haku-form-editar-usuario" class="haku-form-interactivo" style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem;">
                    <input type="hidden" id="haku-user-editar-id" value="${u.id || ''}">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nombre de usuario</label>
                            <input type="text" class="input-text" style="width:100%; opacity:0.8;" value="${escapeHtml(u.username || '')}" disabled>
                        </div>
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nombre completo *</label>
                            <input type="text" id="haku-user-editar-nombre" required class="input-text" style="width:100%;" value="${escapeHtml(u.nombre_completo || '')}">
                        </div>
                    </div>
                    <div>
                        <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Rol asignado *</label>
                        <select id="haku-user-editar-rol" required class="input-text" style="width:100%; background:var(--bg-main); color:var(--text-main);">
                            ${rolesHtml}
                        </select>
                    </div>
                    <div id="haku-user-editar-almacenes-bloque" style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:6px; padding:0.75rem;">
                        <label style="display:block; font-size:0.8rem; font-weight:600; color:var(--text-main); margin-bottom:0.4rem;">
                            Asignación de Almacenes
                        </label>
                        <div style="margin-bottom:0.5rem;">
                            <label style="display:flex; align-items:center; gap:0.4rem; font-size:0.85rem; font-weight:500; cursor:pointer;">
                                <input type="checkbox" id="haku-user-editar-todos-almacenes" ${u.acceso_todos_almacenes ? "checked" : ""}>
                                <strong>Acceso a todos los almacenes</strong>
                            </label>
                        </div>
                        <div id="haku-user-editar-lista-almacenes" style="display:flex; flex-direction:column; gap:0.35rem; max-height:140px; overflow-y:auto; padding-left:0.25rem;">
                            ${almacenesCheckboxes}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Estado de la cuenta</label>
                            <select id="haku-user-editar-activo" class="input-text" style="width:100%; background:var(--bg-main); color:var(--text-main);">
                                <option value="true" ${u.is_active ? "selected" : ""}>Activa</option>
                                <option value="false" ${!u.is_active ? "selected" : ""}>Inactiva / Bloqueada</option>
                            </select>
                        </div>
                        <div>
                            <label style="display:block; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.25rem;">Nueva contraseña (opcional)</label>
                            <input type="password" id="haku-user-editar-password" class="input-text" style="width:100%;" placeholder="Dejar en blanco para mantener">
                        </div>
                    </div>
                    <div class="propuesta-acciones" style="margin-top: 0.5rem;">
                        <button type="submit" class="btn btn-confirmar">💾 Guardar cambios</button>
                        <button type="button" class="btn btn-cancelar" id="haku-btn-cancelar-editar-usuario">❌ Cancelar</button>
                    </div>
                </form>
            </div>
        `;
        contenedor.style.display = "block";

        const selRolEdit = document.getElementById("haku-user-editar-rol");
        const bloqueAlmEdit = document.getElementById("haku-user-editar-almacenes-bloque");
        const chkTodosEdit = document.getElementById("haku-user-editar-todos-almacenes");
        const listaAlmEdit = document.getElementById("haku-user-editar-lista-almacenes");

        if (chkTodosEdit && listaAlmEdit) {
            const toggleLista = () => {
                const checkboxes = listaAlmEdit.querySelectorAll(".haku-chk-alm-edit");
                checkboxes.forEach(cb => {
                    cb.disabled = chkTodosEdit.checked;
                    if (chkTodosEdit.checked) cb.checked = false;
                });
            };
            chkTodosEdit.addEventListener("change", toggleLista);
            listaAlmEdit.querySelectorAll(".haku-chk-alm-edit").forEach(cb => {
                cb.addEventListener("change", () => {
                    if (cb.checked && chkTodosEdit.checked) {
                        chkTodosEdit.checked = false;
                        toggleLista();
                    }
                });
            });
            toggleLista();
        }

        if (selRolEdit && bloqueAlmEdit) {
            selRolEdit.addEventListener("change", () => {
                if (selRolEdit.value === "USUARIO") {
                    bloqueAlmEdit.style.display = "block";
                } else {
                    bloqueAlmEdit.style.display = "none";
                }
            });
            selRolEdit.dispatchEvent(new Event("change"));
        }

        document.getElementById("haku-btn-cancelar-editar-usuario")?.addEventListener("click", () => {
            ocultarTarjetaPropuesta();
            agregarMensajeChat("haku", "Modificación cancelada por el usuario.");
        });

        document.getElementById("haku-form-editar-usuario")?.addEventListener("submit", async (e) => {
            e.preventDefault();
            const rolVal = document.getElementById("haku-user-editar-rol").value.trim();
            const accesoTodos = rolVal === "USUARIO" && document.getElementById("haku-user-editar-todos-almacenes").checked;
            const almsSeleccionados = [];
            document.querySelectorAll("#haku-user-editar-lista-almacenes .haku-chk-alm-edit:checked").forEach(cb => {
                almsSeleccionados.push(parseInt(cb.value, 10));
            });

            if (rolVal === "USUARIO" && !accesoTodos && almsSeleccionados.length === 0) {
                mostrarToast("Debes seleccionar al menos un almacén o marcar acceso a todos.", "error");
                return;
            }

            const payload = {
                id: parseInt(document.getElementById("haku-user-editar-id").value, 10),
                nombre_completo: document.getElementById("haku-user-editar-nombre").value.trim(),
                rol: rolVal,
                almacen_asignado_id: almsSeleccionados.length > 0 ? almsSeleccionados[0] : null,
                almacenes_asignados_ids: almsSeleccionados,
                acceso_todos_almacenes: accesoTodos,
                is_active: document.getElementById("haku-user-editar-activo").value === "true",
            };

            const pwd = document.getElementById("haku-user-editar-password").value;
            if (pwd) {
                payload.password = pwd;
            }

            actualizarEstadoVisual("procesando");
            try {
                const resp = await apiFetch(CONFIG.ENDPOINTS.USUARIO_EDITAR, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const resData = await resp.json();
                if (resData.error) {
                    mostrarToast(resData.error, "error");
                    let detalleTexto = "";
                    if (resData.detalles) {
                        detalleTexto = " " + Object.values(resData.detalles).join("; ");
                    }
                    agregarMensajeChat("haku", `⚠️ No se pudo modificar el usuario: ${resData.error}${detalleTexto}`);
                } else {
                    ocultarTarjetaPropuesta();
                    mostrarToast("Usuario actualizado con éxito.", "info");
                    agregarMensajeChat("haku", `✅ ${resData.mensaje_pantalla}`);
                    if (resData.mensaje_verbal) reproducirVozCliente(resData.mensaje_verbal);
                }
            } catch (err) {
                console.error("Error al editar usuario:", err);
                mostrarToast("Error de conexión al modificar usuario.", "error");
            } finally {
                actualizarEstadoVisual("reposo");
            }
        });
    }

    function ocultarTarjetaPropuesta() {
        const contenedor = document.getElementById("haku-propuesta-card");
        if (contenedor) {
            contenedor.style.display = "none";
            contenedor.innerHTML = "";
        }
    }

    function mostrarToast(mensaje, tipo = "info") {
        const contenedorToast = document.getElementById("haku-toasts") || document.body;
        const toast = document.createElement("div");
        toast.className = `haku-toast haku-toast-${tipo}`;
        toast.textContent = mensaje;
        contenedorToast.appendChild(toast);

        setTimeout(() => {
            toast.classList.add("haku-toast-fadeout");
            setTimeout(() => toast.remove(), 400);
        }, 4000);
    }

    function escapeHtml(texto) {
        if (!texto) return "";
        return String(texto)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ENVÍO DE MENSAJE MANUAL POR TECLADO
    // ─────────────────────────────────────────────────────────────────────────

    function configurarInputManual() {
        const inputTexto = document.getElementById("haku-input-texto");
        const btnEnviar = document.getElementById("haku-btn-enviar");

        if (!inputTexto || !btnEnviar) return;

        function enviar() {
            if (state.procesando) return;
            const val = inputTexto.value.trim();
            if (!val) return;
            inputTexto.value = "";
            agregarMensajeChat("usuario", val);
            enviarComandoHaku(val);
        }

        btnEnviar.addEventListener("click", (e) => {
            e.preventDefault();
            enviar();
        });

        inputTexto.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                enviar();
            }
        });
    }

    function configurarAccesosRapidos() {
        document.querySelectorAll("[data-haku-comando]").forEach((boton) => {
            boton.addEventListener("click", () => {
                const comando = (boton.dataset.hakuComando || "").trim();
                if (comando) enviarComandoHaku(comando);
            });
        });

        const btnCrear = document.getElementById("haku-btn-abrir-crear");
        const dataEl = document.getElementById("haku-crear-producto-datos");
        if (btnCrear && dataEl) {
            btnCrear.addEventListener("click", () => {
                try {
                    const datos = JSON.parse(dataEl.textContent || "{}");
                    mostrarFormularioCrearProducto(datos);
                } catch (err) {
                    console.error("[HakuVoz] Error al abrir formulario de creación:", err);
                }
            });
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // INICIALIZACIÓN
    // ─────────────────────────────────────────────────────────────────────────

    function verificarDisponibilidadMotorVoz(intento = 0) {
        if (!CONFIG.ENDPOINTS.SALUD || intento > 120) return;
        // Sondeo de disponibilidad en segundo plano con prioridad baja para no competir con interacciones críticas.
        fetch(CONFIG.ENDPOINTS.SALUD, { credentials: "same-origin", priority: "low" })
            .then((res) => (res.ok ? res.json() : null))
            .then((data) => {
                if (!data) return;
                const label = document.getElementById("haku-status-label");
                if (data.warmup_iniciado && !data.warmup_completado) {
                    if (label && !state.grabando && !state.procesando && !state.reproduciendoAudio) {
                        label.textContent = "Inicializando motor de voz, espera un momento...";
                    }
                    setTimeout(() => verificarDisponibilidadMotorVoz(intento + 1), 3000);
                } else if (intento > 0 && !state.grabando && !state.procesando && !state.reproduciendoAudio) {
                    actualizarEstadoVisual("reposo");
                }
            })
            .catch(() => {});
    }

    function init() {
        configurarPushToTalkTeclado();
        configurarBotonMicrofono();
        configurarInputManual();
        configurarAccesosRapidos();
        reiniciarTimerInactividad();
        verificarDisponibilidadMotorVoz();

        // Consultar estado inicial acotado a la conversación vigente (evita resucitar propuestas de chats previos)
        const estadoUrl = state.conversacionId
            ? `${CONFIG.ENDPOINTS.ESTADO}?conversacion_id=${encodeURIComponent(state.conversacionId)}`
            : CONFIG.ENDPOINTS.ESTADO;
        apiFetch(estadoUrl)
            .then(res => res.json())
            .then(data => {
                if (data.tiene_propuesta_activa && data.propuesta_activa) {
                    state.propuestaActiva = data.propuesta_activa;
                    mostrarTarjetaPropuesta(data.propuesta_activa);
                }
            })
            .catch(() => {});
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

    // Exponer API global de conveniencia
    window.HakuVoz = {
        iniciarGrabacion,
        detenerGrabacion,
        enviarComando: enviarComandoHaku,
        abrirFormularioCrearProducto: mostrarFormularioCrearProducto,
        confirmarPropuesta: () => state.propuestaActiva && confirmarPropuestaManual(state.propuestaActiva.id),
        cancelarPropuesta: () => state.propuestaActiva && cancelarPropuestaManual(state.propuestaActiva.id),
    };

})();
