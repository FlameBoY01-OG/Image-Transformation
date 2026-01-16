let contentFile = null;
let styleFile = null;

document.addEventListener('DOMContentLoaded', function() {
    setupFileInputs();
    setupDragAndDrop();
});

function setupFileInputs() {
    document.getElementById('contentImg').addEventListener('change', function(evt) {
        const file = evt.target.files[0];
        if (file) {
            contentFile = file;
            displayPreview(file, 'contentPreview', 'contentDropZone');
        }
    });

    document.getElementById('styleImg').addEventListener('change', function(evt) {
        const file = evt.target.files[0];
        if (file) {
            styleFile = file;
            displayPreview(file, 'stylePreview', 'styleDropZone');
        }
    });
}

function setupDragAndDrop() {
    const dropZones = document.querySelectorAll('.drop-zone');
    
    dropZones.forEach(zone => {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            zone.addEventListener(eventName, () => {
                zone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, () => {
                zone.classList.remove('dragover');
            }, false);
        });

        zone.addEventListener('drop', handleDrop, false);
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Please drop an image file!');
            return;
        }
        
        const target = e.currentTarget.dataset.target;
        
        if (target === 'content') {
            contentFile = file;
            document.getElementById('contentImg').files = files;
            displayPreview(file, 'contentPreview', 'contentDropZone');
        } else if (target === 'style') {
            styleFile = file;
            document.getElementById('styleImg').files = files;
            displayPreview(file, 'stylePreview', 'styleDropZone');
        }
    }
}

function displayPreview(file, previewId, dropZoneId) {
    const preview = document.getElementById(previewId);
    const dropZone = document.getElementById(dropZoneId);
    const reader = new FileReader();
    
    reader.onload = function(e) {
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
        preview.classList.add('active');
        dropZone.classList.add('has-image');
    };
    
    reader.readAsDataURL(file);
}

async function generateArt() {
    const btn = document.getElementById('generateBtn');
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');

    if (!contentFile || !styleFile) {
        alert("Please upload both images first!");
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Processing...';
    loader.classList.remove('hidden');
    resultSection.style.display = 'none';

    const formData = new FormData();
    formData.append('content', contentFile);
    formData.append('style', styleFile);

    try {
        const response = await fetch('http://localhost:8000/transform', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            const resultImg = document.getElementById('resultImage');
            resultImg.src = imageUrl;
            document.getElementById('downloadLink').href = imageUrl;
            resultSection.style.display = 'block';
            
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            alert("Server Error: Something went wrong with the style transfer. Check console for details.");
        }
    } catch (error) {
        console.error('Connection error:', error);
        alert("Connection Failed: Make sure the Python backend is running on port 8000.");
    } finally {
        btn.disabled = false;
        btn.textContent = 'Transform';
        loader.classList.add('hidden');
    }
}

function reset() {
    contentFile = null;
    styleFile = null;
    
    document.getElementById('contentImg').value = '';
    document.getElementById('styleImg').value = '';
    
    document.getElementById('contentPreview').innerHTML = '';
    document.getElementById('contentPreview').classList.remove('active');
    document.getElementById('contentDropZone').classList.remove('has-image');
    
    document.getElementById('stylePreview').innerHTML = '';
    document.getElementById('stylePreview').classList.remove('active');
    document.getElementById('styleDropZone').classList.remove('has-image');
    
    document.getElementById('resultSection').style.display = 'none';
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}