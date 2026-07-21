// src/router.js
// Maps a file's extension to the correct backend URL.

const BACKEND_MAP = {
  wav: process.env.REACT_APP_VOICE_BACKEND_URL,
  mp3: process.env.REACT_APP_VOICE_BACKEND_URL,
  docx: process.env.REACT_APP_TEXT_BACKEND_URL,
  pdf: process.env.REACT_APP_TEXT_BACKEND_URL,
  txt: process.env.REACT_APP_TEXT_BACKEND_URL,
  jpg: process.env.REACT_APP_IMAGE_BACKEND_URL,
  jpeg: process.env.REACT_APP_IMAGE_BACKEND_URL,
  png: process.env.REACT_APP_IMAGE_BACKEND_URL,
  webp: process.env.REACT_APP_IMAGE_BACKEND_URL,
  bmp: process.env.REACT_APP_IMAGE_BACKEND_URL,
};

export function getBackendUrl(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const url = BACKEND_MAP[ext];
  if (!url) {
    throw new Error(`Unsupported file type: .${ext}`);
  }
  return url;
}
