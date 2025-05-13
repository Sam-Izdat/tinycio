function createImageBlender({ id, imageSources, baseImage = null, globalWeight = 0.5, labels = null }) {
    const container = document.getElementById(id);
    if (!container) return;

    container.innerHTML = `
        <canvas id="${id}-canvas"></canvas>
        <div id="${id}-sliders" class="blender-sliders"></div>
    `;

    const canvas = document.getElementById(`${id}-canvas`);
    const ctx = canvas.getContext("2d");

    let base = new Image();
    let images = [];
    let sliders = [];
    let loaded = 0;

    function updateBlend() {
        const weights = sliders.map(s => parseFloat(s.value));
        const sum = weights.reduce((a, b) => a + b, 0);
        const normalized = sum > 0 ? weights.map(w => w / sum) : weights;

        // Draw base first
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1;
        ctx.drawImage(base, 0, 0, canvas.width, canvas.height);

        // Blend noise images additively using globalWeight
        const offscreenCanvas = document.createElement("canvas");
        const offscreenCtx = offscreenCanvas.getContext("2d");
        offscreenCanvas.width = canvas.width;
        offscreenCanvas.height = canvas.height;

        normalized.forEach((weight, i) => {
            offscreenCtx.globalAlpha = weight;
            offscreenCtx.drawImage(images[i], 0, 0, canvas.width, canvas.height);
        });

        // Apply additive noise transformation (avoiding getImageData)
        ctx.globalCompositeOperation = "lighter"; // Adds pixel values
        ctx.globalAlpha = globalWeight;
        ctx.drawImage(offscreenCanvas, 0, 0);
        ctx.globalCompositeOperation = "source-over"; // Reset mode
    }

    function setupSlider(index, label, defaultValue = "0") {
        const container = document.createElement("div");
        container.className = "slider-container";

        const sliderLabel = document.createElement("label");
        sliderLabel.innerText = label || `Image ${index + 1}`;
        sliderLabel.htmlFor = `${id}-slider-${index}`;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = "1";
        slider.step = "0.01";
        slider.value = defaultValue;
        slider.id = `${id}-slider-${index}`;
        slider.addEventListener("input", updateBlend);

        container.appendChild(sliderLabel);
        container.appendChild(slider);
        return container;
    }

    // Global Weight Slider
    const gwContainer = document.createElement("div");
    gwContainer.className = "slider-container";
    const gwLabel = document.createElement("label");
    gwLabel.innerText = "Weight";
    gwLabel.htmlFor = `${id}-slider-global-weight`;
    const gwSlider = document.createElement("input");
    gwSlider.type = "range";
    gwSlider.min = "0";
    gwSlider.max = "1";
    gwSlider.step = "0.01";
    gwSlider.value = globalWeight.toString();
    gwSlider.id = `${id}-slider-global-weight`;
    gwSlider.addEventListener("input", (e) => {
        globalWeight = parseFloat(e.target.value);
        updateBlend();
    });
    gwContainer.appendChild(gwLabel);
    gwContainer.appendChild(gwSlider);
    document.getElementById(`${id}-sliders`).appendChild(gwContainer);

    // Load Base
    if (baseImage) {
        base.src = baseImage;
        base.onload = () => {
            loaded++;
            checkReady();
        };
    } else {
        loaded++; // No base image
    }

    // Load Images
    imageSources.forEach((src, i) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
            loaded++;
            checkReady();
        };
        images.push(img);

        const slider = setupSlider(i, labels ? labels[i] : null, i === 0 ? "1" : "0");
        sliders.push(slider.querySelector("input"));
        document.getElementById(`${id}-sliders`).appendChild(slider);
    });

    function checkReady() {
        if (loaded === images.length + (baseImage ? 1 : 0)) {
            canvas.width = base.width || images[0].width;
            canvas.height = base.height || images[0].height;
            updateBlend();
        }
    }
}
