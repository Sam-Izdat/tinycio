function createSlideViewer({ 
    id, 
    imageSrc, 
    slideWidth, 
    slideHeight, 
    hOffset = 0, 
    vOffset = 0, 
    frames = 10, 
    direction = 'H',
    labels = [],
    padLeft = 0,  // Padding before each tile
    padRight = 0  // Padding after each tile
}) {
    const container = document.getElementById(id);
    if (!container) return;

    // Create the HTML structure
    container.innerHTML = `
        <div class="carousel-viewer-container" style="display: flex; flex-direction: column; align-items: center;">
            <div class="carousel-sprite-viewer" style="width: ${slideWidth}px; height: ${slideHeight}px; overflow: hidden; position: relative;">
                <img id="${id}-sprite" src="${imageSrc}" 
                    style="position: absolute; left: -${hOffset}px; top: -${vOffset}px; width: auto; height: auto; max-width: none; max-height: none;">
            </div>
            <div id="${id}-label" style="margin: 5px 0; font-weight: bold;">${labels[0] || ''}</div>
            <input id="${id}-slider" type="range" min="0" max="${frames - 1}" step="1" value="0" style="width: 100%;">
        </div>
    `;

    const img = document.getElementById(`${id}-sprite`);
    const slider = document.getElementById(`${id}-slider`);
    const label = document.getElementById(`${id}-label`);

    slider.addEventListener('input', () => {
        const frame = parseInt(slider.value, 10);
        const offset = frame * (direction === 'H' 
            ? slideWidth + padLeft + padRight
            : slideHeight + padLeft + padRight);
        
        if (direction === 'H') {
            img.style.transform = `translateX(-${hOffset + offset + padLeft}px)`;
        } else {
            img.style.transform = `translateY(-${vOffset + offset + padLeft}px)`;
        }

        if (labels.length) {
            label.textContent = labels[frame] || '';
        }
    });
}
