(() => {
  const fileInput = document.getElementById('media-file');
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const label = document.getElementById('file-name');
      label.textContent = fileInput.files.length ? fileInput.files[0].name : 'Choose an image or video';
    });
    document.getElementById('upload-form').addEventListener('submit', () => {
      const button = document.getElementById('upload-submit');
      button.disabled = true;
      button.textContent = 'Processing locally…';
    });
  }

  const video = document.getElementById('camera-video');
  if (video) {
    const startButton = document.getElementById('camera-start');
    const captureButton = document.getElementById('camera-capture');
    const submitButton = document.getElementById('camera-submit');
    const canvas = document.getElementById('camera-canvas');
    let stream = null;

    startButton.addEventListener('click', async () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
        video.srcObject = null;
        startButton.textContent = 'Start webcam';
        captureButton.disabled = true;
        document.getElementById('camera-placeholder').hidden = false;
        return;
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        video.srcObject = stream;
        document.getElementById('camera-placeholder').hidden = true;
        startButton.textContent = 'Stop webcam';
        captureButton.disabled = false;
      } catch (_error) {
        window.alert('Camera access is unavailable or was not allowed. You can still upload an image or video.');
      }
    });

    captureButton.addEventListener('click', () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      document.getElementById('webcam-image').value = canvas.toDataURL('image/jpeg', 0.88);
      submitButton.disabled = false;
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
      video.srcObject = null;
      startButton.textContent = 'Start webcam';
      captureButton.disabled = true;
      document.getElementById('camera-placeholder').textContent = 'Frame captured and ready for local analysis';
      document.getElementById('camera-placeholder').hidden = false;
    });

    window.addEventListener('pagehide', () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
    });
  }

})();