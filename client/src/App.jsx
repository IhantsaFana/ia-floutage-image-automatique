// src/App.jsx
import React, { useState, useRef } from "react";
import axios from "axios";
import { Upload, ArrowRight, CheckCircle, XCircle, Loader } from "lucide-react";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImg, setResultImg] = useState(null);
  const [floutage, setFloutage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [notif, setNotif] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Quand l'utilisateur sélectionne une image
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResultImg(null);
    setFloutage(null);
    setNotif(null);
    setError(null);
  };

  // Déclenchement du dialogue de sélection de fichier
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Zone de drop pour le drag & drop
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResultImg(null);
      setFloutage(null);
      setNotif(null);
      setError(null);
    }
  };

  // Prévention du comportement par défaut
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  // Envoi à l'API Django
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    
    setLoading(true);
    setResultImg(null);
    setFloutage(null);
    setNotif(null);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      
      // Mets ici l'URL de ton backend Django
      const res = await axios.post("http://localhost:8000/api/detect/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      
      setFloutage(res.data.floutage);
      setResultImg("data:image/jpeg;base64," + res.data.image_base64);
      
      if (res.data.floutage === true) {
        setNotif("Marque sucrée détectée : floutage appliqué !");
      } else if (res.data.floutage === false) {
        setNotif("Pas de floutage : marque non sucrée !");
      } else {
        setNotif(null);
      }
    } catch (err) {
      setError(
        err.response?.data?.error ||
        err.message ||
        "Erreur lors de l'envoi."
      );
    }
    
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-3xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Détection & Floutage de Marque</h1>
          <p className="text-gray-600 mt-2">
            Téléchargez une image pour détecter et flouter automatiquement les marques sucrées
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <form onSubmit={handleSubmit}>
            <div 
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
              onClick={triggerFileInput}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                ref={fileInputRef}
              />
              
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              
              <div className="mt-4 text-gray-700">
                <p className="font-medium">Cliquez pour sélectionner une image</p>
                <p className="text-sm text-gray-500 mt-1">ou glissez-déposez votre fichier ici</p>
              </div>
              
              {selectedFile && (
                <p className="mt-2 text-sm text-blue-600">
                  {selectedFile.name} ({Math.round(selectedFile.size / 1024)} Ko)
                </p>
              )}
            </div>

            <div className="mt-6 flex justify-center">
              <button
                type="submit"
                disabled={!selectedFile || loading}
                className={`px-6 py-2 rounded-md flex items-center justify-center space-x-2 ${
                  !selectedFile || loading
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                } transition-colors`}
              >
                {loading ? (
                  <>
                    <Loader className="h-5 w-5 animate-spin" />
                    <span>Traitement en cours...</span>
                  </>
                ) : (
                  <>
                    <ArrowRight className="h-5 w-5" />
                    <span>Analyser l'image</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {(preview || resultImg) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {preview && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <h2 className="text-lg font-medium text-gray-800 mb-3">Image originale</h2>
                <div className="aspect-auto">
                  <img 
                    src={preview} 
                    alt="Prévisualisation" 
                    className="w-full h-auto rounded-md object-contain max-h-80" 
                  />
                </div>
              </div>
            )}

            {resultImg && (
              <div className="bg-white rounded-lg shadow-md p-4">
                <h2 className="text-lg font-medium text-gray-800 mb-3">Résultat traité</h2>
                <div className="aspect-auto">
                  <img 
                    src={resultImg} 
                    alt="Résultat" 
                    className="w-full h-auto rounded-md object-contain max-h-80" 
                  />
                </div>
                
                {notif && (
                  <div className={`mt-4 p-3 rounded-md flex items-center ${
                    floutage ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"
                  }`}>
                    {floutage ? 
                      <XCircle className="h-5 w-5 mr-2 flex-shrink-0" /> : 
                      <CheckCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                    }
                    <span className="text-sm font-medium">{notif}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-md p-4">
            <div className="flex">
              <XCircle className="h-5 w-5 text-red-600 mr-2" />
              <div>
                <h3 className="text-sm font-medium text-red-800">Une erreur est survenue</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;