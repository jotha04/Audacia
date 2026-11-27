/**
 * Escáner de códigos de barras por cámara en vivo y por foto (imagen subida o capturada)
 * para el formulario web de Creación/Edición de Productos.
 */
(function () {
    "use strict";

    const FORMATOS = [
        "ean_13",
        "ean_8",
        "upc_a",
        "upc_e",
        "code_128",
        "code_39",
        "qr_code",
    ];

    let streamActivo = null;
    let animId = null;
    let detector = null;
    let ultimoCodigo = null;
    let ultimoTiempo = 0;

    function crearDetector() {
        if (!("BarcodeDetector" in window)) return null;
        try {
            return new BarcodeDetector({ formats: FORMATOS });
        } catch (_err) {
            try {
                return new BarcodeDetector();
            } catch (_err2) {
                return null;
            }
        }
    }

    function setEstado(el, texto, tipo) {
        if (!el) return;
        el.style.display = "block";
        el.textContent = texto;
        if (tipo === "ok") el.style.color = "var(--ok, #10b981)";
        else if (tipo === "warn") el.style.color = "var(--warn, #f59e0b)";
        else if (tipo === "error") el.style.color = "var(--danger, #ef4444)";
        else el.style.color = "var(--text-muted, #94a3b8)";
    }

    async function decodificarDesdeImagen(file) {
        const det = crearDetector();
        if (!det) {
            throw new Error("NO_BARCODE_DETECTOR");
        }
        if (typeof createImageBitmap === "function") {
            const bitmap = await createImageBitmap(file);
            try {
                const res = await det.detect(bitmap);
                return (res && res.length && res[0].rawValue) ? res[0].rawValue.trim() : null;
            } finally {
                if (typeof bitmap.close === "function") bitmap.close();
            }
        }
        const imgUrl = URL.createObjectURL(file);
        try {
            const img = await new Promise((resolve, reject) => {
                const el = new Image();
                el.onload = () => resolve(el);
                el.onerror = reject;
                el.src = imgUrl;
            });
            const res = await det.detect(img);
            return (res && res.length && res[0].rawValue) ? res[0].rawValue.trim() : null;
        } finally {
            URL.revokeObjectURL(imgUrl);
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        const root = document.getElementById("web-barcode-widget");
        if (!root) return;

        const inputId = root.dataset.inputId;
        const verifyUrl = root.dataset.verifyUrl || "/api/haku/producto/codigo-barras/";
        const inputCodigo = document.getElementById(inputId);
        const btnCamara = document.getElementById("web-btn-escanear-camara");
        const btnFoto = document.getElementById("web-btn-escanear-foto");
        const inputFoto = document.getElementById("web-input-foto-barcode");
        const btnVerificar = document.getElementById("web-btn-verificar-barcode");
        const msgBox = document.getElementById("web-barcode-msg");

        const visor = document.getElementById("web-scanner-container");
        const video = document.getElementById("web-scanner-video");
        const selectCam = document.getElementById("web-scanner-select-cam");
        const btnCambiarCam = document.getElementById("web-scanner-btn-cambiar");
        const btnCerrar = document.getElementById("web-scanner-btn-cerrar");
        const statusVisor = document.getElementById("web-scanner-status");

        if (!inputCodigo) return;

        function detenerCamara() {
            if (animId) {
                cancelAnimationFrame(animId);
                animId = null;
            }
            if (streamActivo) {
                try {
                    streamActivo.getTracks().forEach((t) => t.stop());
                } catch (_err) {
                    // noop
                }
                streamActivo = null;
            }
            if (video) video.srcObject = null;
            if (visor) visor.style.display = "none";
        }

        async function verificarCodigo(codigo) {
            const cod = String(codigo || "").trim();
            if (!cod) {
                setEstado(msgBox, "Ingresa o escanea un código de barras para verificar.", "warn");
                return;
            }
            inputCodigo.value = cod;
            setEstado(msgBox, "Verificando código de barras en el catálogo...", "info");
            try {
                const resp = await fetch(`${verifyUrl}?codigo=${encodeURIComponent(cod)}`, {
                    credentials: "same-origin",
                });
                const data = await resp.json();
                if (!data.valido) {
                    setEstado(msgBox, data.error || "Formato de código de barras no estándar.", "error");
                    return;
                }
                if (data.existe && data.producto) {
                    setEstado(
                        msgBox,
                        `Este código ya pertenece a '${data.producto.nombre}' (${data.producto.codigo_interno}).`,
                        "error"
                    );
                    return;
                }
                if (data.encontrado_en_catalogo && data.datos_sugeridos) {
                    const sug = data.datos_sugeridos;
                    const inpNom = document.getElementById(root.dataset.nombreId || "id_nombre");
                    const inpEmp = document.getElementById(root.dataset.empaqueId || "id_nombre_empaque");
                    const inpUds = document.getElementById(root.dataset.udsId || "id_unidades_por_empaque");
                    const inpDesc = document.getElementById(root.dataset.descId || "id_descripcion");
                    if (inpNom && !inpNom.value && sug.nombre) inpNom.value = sug.nombre;
                    if (inpEmp && !inpEmp.value && sug.nombre_empaque) inpEmp.value = sug.nombre_empaque;
                    if (inpUds && !inpUds.value && sug.unidades_por_empaque) inpUds.value = sug.unidades_por_empaque;
                    if (inpDesc && !inpDesc.value && sug.descripcion) inpDesc.value = sug.descripcion;
                    setEstado(msgBox, `Código ${data.formato || ""} válido. Datos sugeridos cargados.`, "ok");
                } else {
                    setEstado(msgBox, `Código ${data.formato || ""} válido y disponible para registrar.`, "ok");
                }
            } catch (_err) {
                setEstado(msgBox, "No se pudo verificar el código en este momento (puedes guardarlo igualmente).", "warn");
            }
        }

        async function iniciarCamara(deviceIdForzado) {
            detenerCamara();
            if (!visor || !video) return;
            visor.style.display = "block";
            setEstado(statusVisor, "Solicitando permisos de cámara...", "info");

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setEstado(statusVisor, "Tu navegador no permite acceso a cámara. Usa «Foto» o escríbelo manualmente.", "warn");
                return;
            }

            detector = crearDetector();

            try {
                const devs = await navigator.mediaDevices.enumerateDevices();
                const cams = devs.filter((d) => d.kind === "videoinput");
                if (selectCam) {
                    const prev = deviceIdForzado || selectCam.value;
                    selectCam.innerHTML = "";
                    cams.forEach((dev, idx) => {
                        const opt = document.createElement("option");
                        opt.value = dev.deviceId;
                        opt.textContent = dev.label || `Cámara ${idx + 1}`;
                        if (dev.deviceId === prev) opt.selected = true;
                        selectCam.appendChild(opt);
                    });
                    if (cams.length === 0) {
                        const opt = document.createElement("option");
                        opt.value = "";
                        opt.textContent = "Cámara por defecto";
                        selectCam.appendChild(opt);
                    }
                }
            } catch (_err) {
                // noop
            }

            const chosenId = deviceIdForzado || (selectCam ? selectCam.value : "");
            const constraints = {
                video: chosenId ? { deviceId: { exact: chosenId } } : { facingMode: "environment" },
                audio: false,
            };

            try {
                streamActivo = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = streamActivo;
                await video.play();

                if (!detector) {
                    setEstado(
                        statusVisor,
                        "Cámara activa, pero tu navegador no soporta BarcodeDetector nativo. Puedes leer el código o usar Chrome/Edge.",
                        "warn"
                    );
                    return;
                }

                setEstado(statusVisor, "Apunta la cámara al código de barras (EAN-13, EAN-8, UPC, Code 128)...", "ok");

                let lastScan = 0;
                const loop = async (ts) => {
                    if (!streamActivo) return;
                    if (ts - lastScan > 150 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                        lastScan = ts;
                        try {
                            const codes = await detector.detect(video);
                            if (codes && codes.length > 0) {
                                const raw = codes[0].rawValue ? codes[0].rawValue.trim() : "";
                                const now = Date.now();
                                if (raw && (raw !== ultimoCodigo || now - ultimoTiempo > 3000)) {
                                    ultimoCodigo = raw;
                                    ultimoTiempo = now;
                                    detenerCamara();
                                    await verificarCodigo(raw);
                                    return;
                                }
                            }
                        } catch (_scanErr) {
                            // ignorar frame individual
                        }
                    }
                    animId = requestAnimationFrame(loop);
                };
                animId = requestAnimationFrame(loop);
            } catch (err) {
                const msg = err && err.name === "NotAllowedError"
                    ? "Permiso de cámara denegado. Usa «Foto» o ingresa el código manualmente."
                    : "No se pudo abrir la cámara. Usa «Foto» o ingresa el código manualmente.";
                setEstado(statusVisor, msg, "error");
            }
        }

        btnCamara?.addEventListener("click", () => iniciarCamara());
        btnCerrar?.addEventListener("click", detenerCamara);
        btnCambiarCam?.addEventListener("click", () => {
            if (!selectCam || selectCam.options.length <= 1) return;
            selectCam.selectedIndex = (selectCam.selectedIndex + 1) % selectCam.options.length;
            iniciarCamara(selectCam.value);
        });
        selectCam?.addEventListener("change", () => iniciarCamara(selectCam.value));

        btnFoto?.addEventListener("click", () => {
            if (!inputFoto) return;
            inputFoto.value = "";
            inputFoto.click();
        });

        inputFoto?.addEventListener("change", async () => {
            const file = inputFoto.files && inputFoto.files[0];
            if (!file) return;
            setEstado(msgBox, "Analizando imagen en busca de código de barras...", "info");
            try {
                const codigo = await decodificarDesdeImagen(file);
                if (codigo) {
                    await verificarCodigo(codigo);
                } else {
                    setEstado(
                        msgBox,
                        "No se detectó ningún código de barras claro en la foto. Intenta con una imagen más nítida o acércalo más.",
                        "warn"
                    );
                }
            } catch (err) {
                if (err && err.message === "NO_BARCODE_DETECTOR") {
                    setEstado(
                        msgBox,
                        "Tu navegador no tiene habilitada la API nativa BarcodeDetector para leer fotos (usa Chrome o Edge).",
                        "warn"
                    );
                } else {
                    setEstado(msgBox, "No se pudo procesar la imagen seleccionada.", "error");
                }
            }
        });

        btnVerificar?.addEventListener("click", () => verificarCodigo(inputCodigo.value));
        window.addEventListener("beforeunload", detenerCamara);
    });
})();
