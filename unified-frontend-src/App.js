import React, { useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { MessageList } from 'react-chat-elements';
import 'react-chat-elements/dist/main.css';
import { getBackendUrl } from './router';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      position: 'left',
      type: 'text',
      text: "Hi! Drop a file below - a .wav/.mp3 (voice), .docx/.pdf/.txt (text), or .jpg/.png (image) - and I'll tell you if it's REAL or AI-GENERATED.",
      date: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messageListRef = useRef(null);

  const addMessage = (msg) => {
    setMessages((prev) => [...prev, { ...msg, date: new Date() }]);
  };

  const handleFile = async (file) => {
    const isImage = file.type.startsWith('image/');
    addMessage({
      position: 'right',
      type: isImage ? 'photo' : 'file',
      text: file.name,
      data: isImage
        ? { uri: URL.createObjectURL(file), status: { click: false } }
        : { uri: '#', status: { click: false } },
      title: file.name,
    });

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const backendUrl = getBackendUrl(file.name); // <- picks the right backend based on file type
      const response = await fetch(`${backendUrl}/predict/`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (result.error) {
        addMessage({
          position: 'left',
          type: 'text',
          text: `❌ ${result.error}`,
        });
      } else {
        const verdict = result.predicted_class.toUpperCase();
        const confidencePct = (result.confidence * 100).toFixed(2);
        const emoji = verdict === 'REAL' || verdict === 'HUMAN' ? '✅' : '⚠️';
        addMessage({
          position: 'left',
          type: 'text',
          text: `${emoji} **${verdict}** - ${confidencePct}% confidence`,
        });
      }
    } catch (err) {
      addMessage({
        position: 'left',
        type: 'text',
        text: `❌ Error: ${err.message}`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      handleFile(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'audio/*': ['.wav', '.mp3'],
      'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
  });

  return (
    <div className="App">
      <header className="chat-header">
        <h1>🛡️ Anti-Spoofing Assistant</h1>
        <p>Text · Voice · Image detection in one place</p>
      </header>

      <div className="chat-window">
        <MessageList
          referance={messageListRef}
          className="message-list"
          lockable={true}
          toBottomHeight={'100%'}
          dataSource={messages}
        />
      </div>

      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
        <input {...getInputProps()} />
        <span className="corner-tl"></span>
        <span className="corner-br"></span>
        {isLoading ? (
          <p>Analyzing...</p>
        ) : isDragActive ? (
          <p>Drop the file here...</p>
        ) : (
          <p>📎 Drag & drop a file here, or click to select (audio, image, or document)</p>
        )}
      </div>
    </div>
  );
}

export default App;
