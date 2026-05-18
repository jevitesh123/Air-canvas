/**
 * Air Canvas - Hand Gesture Drawing Application
 * Frontend JavaScript with improved gesture recognition UI
 */

// ============ Page Initialization ============
window.addEventListener('DOMContentLoaded', function() {
    console.log('🎨 Air Canvas Application Loading...');
    setupColorPicker();
    setupBrushSize();
    startStatisticsUpdate();
    setupKeyboardShortcuts();
    loadGestureInstructions();
    checkApplicationHealth();
});

// ============ Configuration ============
const colorMap = {
    '0': { name: 'Blue', bgr: [255, 0, 0], rgb: 'rgb(0, 0, 255)' },
    '1': { name: 'Green', bgr: [0, 255, 0], rgb: 'rgb(0, 255, 0)' },
    '2': { name: 'Red', bgr: [0, 0, 255], rgb: 'rgb(255, 0, 0)' },
    '3': { name: 'Yellow', bgr: [0, 255, 255], rgb: 'rgb(255, 255, 0)' },
    '4': { name: 'Eraser', bgr: [0, 0, 0], rgb: 'rgb(255, 255, 255)' }
};

// ============ Gesture Instructions ============
function loadGestureInstructions() {
    const instructions = {
        'drawing': '✏️ Move your hand inside the frame to draw',
        'color_change': '🎨 Hover over the top toolbar to pick a color',
        'palm_open': '✋ Open palm to clear canvas (use Clear button)',
        'toolbar': '🖱️ Handy toolbar: move into the top region to select colors'
    };
    
    const container = document.querySelector('.tools-panel');
    if (container) {
        let instructionsHTML = '<h5>🤚 Hand Gestures</h5><div class="gesture-instructions">';
        for (const [key, value] of Object.entries(instructions)) {
            instructionsHTML += `<p class="small mb-2"><strong>${value}</strong></p>`;
        }
        instructionsHTML += '</div><hr>';
        
        // Insert before calibration section
        const calibrationSection = container.querySelector('.calibration-section');
        if (calibrationSection) {
            calibrationSection.parentElement.insertBefore(
                document.createRange().createContextualFragment(instructionsHTML),
                calibrationSection
            );
        }
    }
}

// ============ Color Picker Setup ============
function setupColorPicker() {
    const colorPicker = document.getElementById('colorPicker');
    const colorDisplay = document.getElementById('colorDisplay');
    const colorName = document.getElementById('colorName');
    
    if (!colorPicker) return;
    
    // Initial color matches backend default (Blue index 0)
    updateColorDisplay(0);
    
    colorPicker.addEventListener('change', function() {
        const colorIndex = parseInt(this.value);
        updateColorDisplay(colorIndex);
        
        // Send to backend
        setColor(colorIndex);
    });
    
    colorPicker.addEventListener('input', function() {
        const colorIndex = parseInt(this.value);
        updateColorDisplay(colorIndex);
    });
}

function updateColorDisplay(colorIndex) {
    const colorData = colorMap[colorIndex];
    const colorDisplay = document.getElementById('colorDisplay');
    const colorName = document.getElementById('colorName');
    
    if (colorDisplay) {
        colorDisplay.style.backgroundColor = colorData.rgb;
    }
    if (colorName) {
        colorName.textContent = colorData.name;
    }
}

// ============ Brush Size Setup ============
function setupBrushSize() {
    const brushSize = document.getElementById('brushSize');
    const brushValue = document.getElementById('brushValue');
    
    if (!brushSize) return;
    
    brushSize.addEventListener('input', function() {
        const size = this.value;
        if (brushValue) brushValue.textContent = size;
    });
    
    brushSize.addEventListener('change', function() {
        const size = this.value;
        setBrushSize(size);
    });
}

// ============ Statistics Update ============
function startStatisticsUpdate() {
    setInterval(updateStatistics, 1000); // Update every second
}

function updateStatistics() {
    fetch('/stats')
        .then(res => res.json())
        .then(data => {
            if (data && typeof data === 'object') {
                // Update stroke count
                const strokeCount = document.getElementById('strokeCount');
                if (strokeCount) strokeCount.textContent = data.strokes || 0;
                
                // Update time
                const drawingTime = document.getElementById('drawingTime');
                if (drawingTime) {
                    const seconds = data.time || 0;
                    const minutes = Math.floor(seconds / 60);
                    const secs = seconds % 60;
                    drawingTime.textContent = minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
                }
                
                // Update distance
                const totalDistance = document.getElementById('totalDistance');
                if (totalDistance) {
                    totalDistance.textContent = Number(data.distance || 0).toLocaleString();
                }
                
                // Update strokes per minute
                const strokesPerMin = document.getElementById('strokesPerMin');
                if (strokesPerMin) {
                    strokesPerMin.textContent = data.strokes_per_minute || 0;
                }
                
                // Update average distance
                const avgDistance = document.getElementById('avgDistance');
                if (avgDistance) {
                    avgDistance.textContent = data.avg_distance || 0;
                }
                
                // Update current color
                const colorName = document.getElementById('colorName');
                if (colorName && data.current_color) {
                    const colorIdx = Object.keys(colorMap).find(k => colorMap[k].name.includes(data.current_color));
                    if (colorIdx) updateColorDisplay(colorIdx);
                }
                
                // Update colors used
                updateColorsUsedDisplay(data.color_names || []);
            }
        })
        .catch(error => {
            console.error('Error fetching statistics:', error);
        });
}

function updateColorsUsedDisplay(colorNames) {
    const colorsUsedDiv = document.getElementById('colorsUsed');
    if (!colorsUsedDiv) return;
    
    colorsUsedDiv.innerHTML = '';
    
    if (colorNames && colorNames.length > 0) {
        colorNames.forEach(colorName => {
            const colorElement = document.createElement('div');
            colorElement.style.display = 'inline-flex';
            colorElement.style.alignItems = 'center';
            colorElement.style.gap = '6px';
            colorElement.style.padding = '6px 10px';
            colorElement.style.backgroundColor = '#e9ecef';
            colorElement.style.borderRadius = '6px';
            colorElement.style.fontSize = '13px';
            colorElement.style.fontWeight = '500';
            colorElement.style.marginBottom = '4px';
            
            const colorDot = document.createElement('span');
            colorDot.style.display = 'inline-block';
            colorDot.style.width = '14px';
            colorDot.style.height = '14px';
            colorDot.style.borderRadius = '50%';
            colorDot.style.border = '2px solid #333';
            
            // Find color RGB by name
            const colorData = Object.values(colorMap).find(c => c.name.includes(colorName));
            if (colorData) {
                colorDot.style.backgroundColor = colorData.rgb;
            }
            
            const colorLabel = document.createElement('span');
            colorLabel.textContent = colorName;
            
            colorElement.appendChild(colorDot);
            colorElement.appendChild(colorLabel);
            colorsUsedDiv.appendChild(colorElement);
        });
    } else {
        colorsUsedDiv.textContent = 'None yet';
        colorsUsedDiv.style.fontSize = '12px';
        colorsUsedDiv.style.color = '#999';
    }
}

// ============ Keyboard Shortcuts ============
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(event) {
        // Prevent shortcuts when typing in inputs
        if (event.target.tagName === 'INPUT' && event.target.type === 'text') return;
        
        // C for Clear
        if ((event.key === 'c' || event.key === 'C') && !event.ctrlKey) {
            clearCanvas();
        }
        // S for Save
        else if ((event.key === 's' || event.key === 'S') && !event.ctrlKey) {
            saveDrawing();
        }
        // Ctrl+Z for Undo
        else if ((event.ctrlKey || event.metaKey) && event.key === 'z') {
            event.preventDefault();
            undo();
        }
        // Ctrl+Y for Redo
        else if ((event.ctrlKey || event.metaKey) && event.key === 'y') {
            event.preventDefault();
            redo();
        }
        // Number keys (1-5) for color selection
        else if (event.key >= '1' && event.key <= '5') {
            const colorIndex = parseInt(event.key) - 1;
            const colorPicker = document.getElementById('colorPicker');
            if (colorPicker) {
                // If the option exists in select, value will be set
                colorPicker.value = colorIndex;
                colorPicker.dispatchEvent(new Event('change'));
            } else {
                setColor(colorIndex);
                updateColorDisplay(colorIndex);
            }
        }
    });
}

// ============ API Calls ============

function setColor(colorIndex) {
    fetch('/set-color', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ color: colorIndex })
    })
    .then(res => res.json())
    .then(data => {
        console.log(`Color changed: ${data.color?.name}`);
        playSound('click');
    })
    .catch(error => {
        console.error('Error setting color:', error);
        playSound('error');
    });
}

function setBrushSize(size) {
    fetch('/set-brush-size', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ size: parseInt(size) })
    })
    .then(res => res.json())
    .then(data => {
        console.log(`Brush size: ${size}px`);
    })
    .catch(error => {
        console.error('Error setting brush size:', error);
    });
}

function saveDrawing() {
    fetch('/save', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showNotification('✅ Drawing saved!', 'success');
                playSound('success');
                console.log('Drawing saved:', data.filename);
            } else {
                showNotification('❌ ' + data.message, 'error');
                playSound('error');
            }
        })
        .catch(error => {
            console.error('Error saving drawing:', error);
            showNotification('❌ Error saving drawing', 'error');
            playSound('error');
        });
}

function clearCanvas() {
    if (confirm('🧹 Clear canvas? This cannot be undone immediately (use Undo).')) {
        fetch('/clear', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    showNotification('✅ Canvas cleared!', 'success');
                    playSound('success');
                } else {
                    showNotification('❌ ' + data.message, 'error');
                    playSound('error');
                }
            })
            .catch(error => {
                console.error('Error clearing canvas:', error);
                showNotification('❌ Error clearing canvas', 'error');
                playSound('error');
            });
    }
}

function undo() {
    fetch('/undo', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success' || data.status === 'warning') {
                showNotification(data.message, data.status === 'success' ? 'success' : 'info');
                playSound('success');
            } else {
                showNotification(data.message, 'error');
                playSound('error');
            }
        })
        .catch(error => {
            console.error('Error undoing:', error);
            playSound('error');
        });
}

function redo() {
    fetch('/redo', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success' || data.status === 'warning') {
                showNotification(data.message, data.status === 'success' ? 'success' : 'info');
                playSound('success');
            } else {
                showNotification(data.message, 'error');
                playSound('error');
            }
        })
        .catch(error => {
            console.error('Error redoing:', error);
            playSound('error');
        });
}

function takeScreenshot() {
    if (typeof html2canvas !== 'undefined') {
        const target = document.querySelector('.canvas-wrapper');
        if (!target) {
            showNotification('❌ Screenshot target not found', 'error');
            return;
        }

        html2canvas(target).then(canvas => {
            const link = document.createElement('a');
            link.download = `aircanvas_${new Date().getTime()}.png`;
            link.href = canvas.toDataURL();
            link.click();
            showNotification('📸 Screenshot saved!', 'success');
        }).catch(error => {
            console.error('Error taking screenshot:', error);
            showNotification('❌ Error taking screenshot', 'error');
        });
    } else {
        showNotification('⚠️ html2canvas library not loaded', 'warning');
    }
}

function toggleTheme() {
    const isDark = document.body.classList.contains('dark-mode');
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
    showNotification(isDark ? '☀️ Light mode' : '🌙 Dark mode', 'info');
}

// ============ Utility Functions ============

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.padding = '15px 20px';
    notification.style.borderRadius = '8px';
    notification.style.zIndex = '9999';
    notification.style.maxWidth = '300px';
    notification.style.wordWrap = 'break-word';
    notification.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    notification.style.animation = 'slideIn 0.3s ease-out';
    
    const typeStyles = {
        'success': { bg: '#d4edda', color: '#155724', border: '#c3e6cb' },
        'error': { bg: '#f8d7da', color: '#721c24', border: '#f5c6cb' },
        'warning': { bg: '#fff3cd', color: '#856404', border: '#ffeeba' },
        'info': { bg: '#d1ecf1', color: '#0c5460', border: '#bee5eb' }
    };
    
    const style = typeStyles[type] || typeStyles['info'];
    notification.style.backgroundColor = style.bg;
    notification.style.color = style.color;
    notification.style.border = `1px solid ${style.border}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function playSound(type) {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        const soundConfig = {
            'click': { freq: 800, duration: 0.1 },
            'success': { freq: 1200, duration: 0.15 },
            'error': { freq: 400, duration: 0.2 }
        };
        
        const config = soundConfig[type] || soundConfig['click'];
        oscillator.frequency.value = config.freq;
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + config.duration);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + config.duration);
    } catch (e) {
        console.warn('Audio playback not available:', e);
    }
}

function checkApplicationHealth() {
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'healthy') {
                console.log('✅ Application healthy');
            }
        })
        .catch(error => {
            console.warn('⚠️ Health check failed:', error);
            showNotification('⚠️ Connection issue - check if server is running', 'warning');
        });
}

// ============ CSS Animations (Injected) ============
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
    
    .gesture-instructions {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #007bff;
        margin-bottom: 15px;
    }
    
    .gesture-instructions p {
        margin: 0;
        line-height: 1.6;
    }
`;
document.head.appendChild(style);

console.log('✅ Air Canvas Frontend Ready');
