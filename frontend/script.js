// 1. Handle Content Image Preview
document.getElementById('contentImg').onchange = evt => {
    const [file] = document.getElementById('contentImg').files
    if (file) {
        document.getElementById('contentPreview').innerHTML = `<img src="${URL.createObjectURL(file)}" />`
    }
}

// 2. Handle Style Image Preview
document.getElementById('styleImg').onchange = evt => {
    const [file] = document.getElementById('styleImg').files
    if (file) {
        document.getElementById('stylePreview').innerHTML = `<img src="${URL.createObjectURL(file)}" />`
    }
}

// 3. The Main Function
async function generateArt() {
    const contentInput = document.getElementById('contentImg').files[0];
    const styleInput = document.getElementById('styleImg').files[0];
    const btn = document.getElementById('generateBtn');
    const loader = document.getElementById('loader');
    const resultImg = document.getElementById('resultImage');
    const resultSection = document.getElementById('resultSection');

    // Validation
    if (!contentInput || !styleInput) {
        alert("Please upload both images first!");
        return;
    }

    // Lock UI while processing
    btn.disabled = true;
    btn.innerText = "Painting...";
    loader.classList.remove('hidden');
    resultSection.style.display = 'none';

    // Prepare data for upload
    const formData = new FormData();
    formData.append('content', contentInput);
    formData.append('style', styleInput);

    try {
        // Send to Backend
        // Note: We use 'localhost:8000' because that is where Python is listening
        const response = await fetch('http://localhost:8000/transform', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            // Convert response to an image URL
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            // Display Result
            resultImg.src = imageUrl;
            document.getElementById('downloadLink').href = imageUrl;
            resultSection.style.display = 'block';
        } else {
            alert("Server Error: Something went wrong with the style transfer.");
        }
    } catch (error) {
        console.error(error);
        alert("Connection Failed: Is the Python backend running?");
    } finally {
        // Unlock UI
        btn.disabled = false;
        btn.innerText = "Paint It";
        loader.classList.add('hidden');
    }
}