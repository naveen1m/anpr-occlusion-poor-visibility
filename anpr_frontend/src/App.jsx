import React, { useState } from "react";

const ANPRApp = () => {
  const [photo1, setPhoto1] = useState(null);
  const [photo2, setPhoto2] = useState(null);
  const [showResults, setShowResults] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const api_url = " http://127.0.0.1:8000";
  // const api_url = "https://anpr-occlusion-poor-visibility-fastapi.onrender.com";

  const handlePhotoChange = (e, photoSetter) => {
    const file = e.target.files[0];
    if (file) {
      photoSetter(URL.createObjectURL(file));
      setShowResults(false);
    }
  };

  const handleProceed = async () => {
    if (!photo1) {
      alert("Please upload at vehicle photo 1 before proceeding.");
      return;
    }
    setIsLoading(true);
    setShowResults(false);
    const fileInputs = document.querySelectorAll('input[type="file"]');
    const file1 = fileInputs[0]?.files[0];
    const file2 = fileInputs[1]?.files[0];

    const toBase64 = (file) =>
      new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(",")[1]); // Remove base64 prefix
        reader.onerror = (error) => reject(error);
      });

    try {
      const base64Image1 = file1 ? await toBase64(file1) : null;
      const base64Image2 = file2 ? await toBase64(file2) : null;

      const images = [{ image_base64: base64Image1 }];
      if (base64Image2) images.push({ image_base64: base64Image2 });

      const payload = { images };

      const response = await fetch(`${api_url}/predict_number_plate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (!response.ok) {
        setIsLoading(false);
        throw new Error(result.detail || "Failed to process images");
      }

      console.log("API Result:", result);

      setIsLoading(false);
      setResultData(result);
      setShowResults(true);
    } catch (error) {
      setIsLoading(false);
      console.error("Error:", error);
      alert("Error while sending images to backend.");
    }
  };
  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 to-white flex flex-col items-center justify-center p-2">
      <h1 className="text-5xl font-bold text-indigo-700 mb-12 tracking-wide drop-shadow-sm text-center">
        Automated Number Plate Recognition System
        <br />
        <span className="text-4xl">Under Occlusion and Low Visibility</span>
      </h1>
      {/* <h3>Under Occlusion and Low Visibility</h3> */}

      <div className="flex flex-col md:flex-row space-y-6 md:space-y-0 md:space-x-14 mb-3">
        <div className="bg-indigo-100 rounded-xl p-6 flex flex-col items-center space-y-4 shadow-md">
          <label className="text-indigo-800 font-semibold text-md">
            Upload Vehicle Photo 1
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handlePhotoChange(e, setPhoto1)}
            className="block w-48 text-sm text-indigo-700 border border-indigo-500 rounded-lg cursor-pointer bg-white hover:bg-indigo-100 shadow-sm"
          />
          {photo1 && (
            <img
              src={photo1}
              alt="Preview 1"
              className="w-44 h-auto mt-2 border rounded-lg shadow-md"
            />
          )}
        </div>

        <div className="bg-indigo-100 rounded-xl p-6 flex flex-col items-center space-y-4 shadow-md">
          <label className="text-indigo-800 font-semibold text-md">
            Upload Vehicle Photo 2
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => handlePhotoChange(e, setPhoto2)}
            className="block w-48 text-sm text-indigo-700 border border-indigo-500 rounded-lg cursor-pointer bg-white hover:bg-indigo-100 shadow-sm"
          />
          {photo2 && (
            <img
              src={photo2}
              alt="Preview 2"
              className="w-44 h-auto mt-2 border rounded-lg shadow-md"
            />
          )}
        </div>
      </div>

      <div className="mb-4 px-10 py-3 w-36 bg-indigo-600 text-white text-lg font-medium rounded-lg hover:bg-indigo-700 transition duration-200 shadow-md">
        {isLoading ? (
          <Loader />
        ) : (
          <button onClick={handleProceed}>Proceed</button>
        )}
      </div>

      {showResults && (
        <div className="flex flex-col items-center space-y-4">
          <div className="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-14">
            {photo1 && (
              <div className="border border-indigo-400 rounded-lg p-6 text-center bg-white shadow-md w-72">
                <h3 className="text-lg font-semibold text-indigo-700 mb-3">
                  Photo 1 Results
                </h3>
                <p className="text-indigo-700 text-sm">
                  Extracted Text:{" "}
                  <span className="text-amber-700">
                    {resultData.detected_number1}
                  </span>
                </p>
                <p className="text-indigo-700 text-sm">
                  Visibility:
                  <span className="text-amber-700">
                    {" "}
                    {resultData.visibility1}%{" "}
                  </span>
                </p>
              </div>
            )}
            {photo2 && (
              <div className="border border-indigo-400 rounded-lg p-6 text-center bg-white shadow-md w-72">
                <h3 className="text-lg font-semibold text-indigo-700 mb-3">
                  Photo 2 Results
                </h3>
                <p className="text-indigo-700 text-sm">
                  Extracted Text:{" "}
                  <span className="text-amber-700">
                    {resultData.detected_number2}
                  </span>
                </p>
                <p className="text-indigo-700 text-sm">
                  Visibility:{" "}
                  <span className="text-amber-700">
                    {resultData.visibility2}%
                  </span>
                </p>
              </div>
            )}
          </div>
          {photo2 && (
            <div className="border border-indigo-600 rounded-xl p-6 text-center bg-white shadow-lg w-80">
              <h2 className="text-xl font-semibold text-indigo-800 mb-2">
                Final Result
              </h2>
              <p className="text-indigo-900 text-lg font-medium">
                Matched Text:{" "}
                <span className="text-emerald-700 underline underline-offset-2 ">
                  {resultData.predicted_number_plate}{" "}
                </span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
const Loader = () => (
  <>
    <div className="flex flex-col items-center justify-center space-y-3">
      <div className="w-7 h-7 border-4 border-indigo-300 border-t-indigo-600 rounded-full animate-spin"></div>
    </div>
  </>
);
export default ANPRApp;
