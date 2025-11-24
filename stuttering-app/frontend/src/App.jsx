import React, { useState, useRef } from "react";
import {
  Mic,
  Square,
  RotateCcw,
  Activity,
  FileText,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";

const SpeechTherapistApp = () => {
  // State for recording
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioChunksRef = useRef([]);

  // State for API interaction
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const [currentPrompt, setCurrentPrompt] = useState(0);

  const prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "Please pack the box with five dozen liquor jugs.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "I want to go to the store to buy some bread and butter.",
  ];

  // --- RECORDING FUNCTIONS ---

  const startRecording = async () => {
    setError(null);
    setResults(null);
    audioChunksRef.current = []; // Clear previous chunks

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = sendAudioToBackend;

      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
    } catch (err) {
      setError(
        "Microphone access denied. Please allow microphone permissions."
      );
      console.error("Microphone Error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && recording) {
      mediaRecorder.stop(); // This triggers onstop -> sendAudioToBackend
      setRecording(false);
      // Stop all tracks to release microphone
      mediaRecorder.stream.getTracks().forEach((track) => track.stop());
    }
  };

  // --- API INTERACTION ---

  const sendAudioToBackend = async () => {
    setAnalyzing(true);
    const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });

    const formData = new FormData();
    // Filename 'recording.webm' helps backend know format, but pydub handles it anyway
    formData.append("file", audioBlob, "recording.webm");

    try {
      // REPLACE 'http://localhost:8000' with your actual backend URL if different
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Server failed to process audio");
      }

      setResults(data);
    } catch (err) {
      console.error("API Error:", err);
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 p-4 md:p-8">
      <div className="w-full max-w-6xl mx-auto">
        {/* Header */}
        <header className="mb-6 md:mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 md:gap-0">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-indigo-700 flex items-center gap-2">
              <img src="voice.png" alt="FluencyFlow Logo" className="h-10 w-10" />
              FluencyFlow
            </h1>
            <p className="text-sm md:text-base text-slate-500">
              AI-Powered Speech Therapy Assistant
            </p>
          </div>
          <div className="bg-white px-4 py-2 rounded-full shadow-sm border border-slate-200 self-start md:self-auto">
            <span className="text-xs md:text-sm font-semibold text-slate-600">
              User: Final Year Student
            </span>
          </div>
        </header>

        {/* ERROR ALERT BANNER */}
        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-md flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
            <div>
              <h3 className="text-red-800 font-bold text-sm">
                Analysis Failed
              </h3>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel: Practice Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Prompt Card */}
            <div className="bg-white rounded-2xl p-6 md:p-8 shadow-lg border border-slate-100 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-purple-500"></div>
              <h2 className="text-xs md:text-sm font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                <FileText className="w-4 h-4" /> Reading Prompt
              </h2>
              <p className="text-xl md:text-3xl font-medium text-slate-800 leading-relaxed">
                "{prompts[currentPrompt]}"
              </p>
              <div className="mt-6 flex gap-2">
                <button
                  onClick={() =>
                    setCurrentPrompt((prev) => (prev + 1) % prompts.length)
                  }
                  className="text-indigo-600 text-sm font-semibold hover:underline flex items-center"
                >
                  Next Prompt <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Recording Controls */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 flex flex-col items-center justify-center min-h-[250px]">
              {/* STATE: IDLE */}
              {!recording && !analyzing && !results && (
                <div className="text-center">
                  <button
                    onClick={startRecording}
                    className="w-20 h-20 bg-indigo-600 hover:bg-indigo-700 rounded-full flex items-center justify-center shadow-xl shadow-indigo-200 transition-all transform hover:scale-105 active:scale-95"
                  >
                    <Mic className="w-8 h-8 text-white" />
                  </button>
                  <p className="mt-4 text-slate-600 font-medium">
                    Tap to Start Recording
                  </p>
                </div>
              )}

              {/* STATE: RECORDING */}
              {recording && (
                <div className="text-center animate-pulse">
                  <button
                    onClick={stopRecording}
                    className="w-20 h-20 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center shadow-xl shadow-red-200 transition-all active:scale-95"
                  >
                    <Square className="w-8 h-8 text-white fill-current" />
                  </button>
                  <p className="mt-4 text-red-500 font-medium">
                    Recording... (Tap to Stop)
                  </p>
                  <div className="mt-2 flex gap-1 justify-center h-4 items-end">
                    {[...Array(5)].map((_, i) => (
                      <div
                        key={i}
                        className="w-1 bg-red-400 rounded-full animate-bounce"
                        style={{
                          height: Math.random() * 20 + 10 + "px",
                          animationDelay: i * 0.1 + "s",
                        }}
                      ></div>
                    ))}
                  </div>
                </div>
              )}

              {/* STATE: ANALYZING */}
              {analyzing && (
                <div className="text-center">
                  <div className="w-20 h-20 bg-white border-4 border-indigo-100 rounded-full flex items-center justify-center mx-auto relative">
                    <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
                  </div>
                  <p className="mt-4 text-indigo-600 font-medium">
                    Analyzing Speech Patterns...
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Processing audio on local GPU...
                  </p>
                </div>
              )}

              {/* STATE: RESULTS */}
              {results && !analyzing && (
                <div className="w-full animate-in fade-in duration-500">
                  <div className="flex justify-between items-center mb-4 border-b pb-2">
                    <h3 className="font-bold text-slate-700">
                      Analysis Result
                    </h3>
                    <button
                      onClick={() => setResults(null)}
                      className="text-sm text-slate-500 hover:text-indigo-600 flex items-center gap-1"
                    >
                      <RotateCcw className="w-4 h-4" /> Reset
                    </button>
                  </div>

                  {/* Visual Transcript */}
                  <div className="bg-slate-50 rounded-lg p-4 border border-slate-200 mb-4">
                    <h4 className="text-xs font-bold text-slate-400 uppercase mb-2">
                      Transcript & Stutters
                    </h4>
                    <p className="text-lg text-slate-700 leading-relaxed">
                      {results.original_transcript ? (
                        results.original_transcript
                      ) : (
                        <span className="italic text-slate-400">
                          Audio processed, but transcription failed.
                        </span>
                      )}
                    </p>
                    {/* Render Stutter Tags */}
                    {results.events.length > 0 ? (
                      <div className="mt-4 flex flex-wrap gap-2">
                        {results.events.map((event, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 text-xs font-bold rounded bg-red-100 text-red-700 border border-red-200"
                          >
                            {event.type} @ {event.start.toFixed(1)}s
                          </span>
                        ))}
                      </div>
                    ) : (
                      <div className="mt-4 flex items-center gap-2 text-green-600 text-sm font-medium">
                        <CheckCircle className="w-4 h-4" /> No stutters
                        detected.
                      </div>
                    )}
                  </div>

                  {/* Correction */}
                  <div className="bg-indigo-50 rounded-lg p-4 border border-indigo-100">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase mb-1">
                      Corrected Text
                    </h4>
                    <p className="text-indigo-800">
                      {results.corrected_transcript}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: Metrics & Feedback */}
          <div className="space-y-6">
            {/* Score Card */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
              <h3 className="text-slate-500 font-bold text-xs uppercase tracking-wider mb-2">
                Session Fluency Score
              </h3>
              <div className="flex items-end gap-2">
                <span
                  className={`text-5xl md:text-6xl font-bold ${
                    results
                      ? results.fluency_score > 80
                        ? "text-green-500"
                        : "text-yellow-500"
                      : "text-slate-200"
                  }`}
                >
                  {results ? results.fluency_score : "--"}
                </span>
                <span className="text-xl text-slate-400 mb-2">/ 100</span>
              </div>
              <p className="text-xs text-slate-400 mt-2">
                {results
                  ? "Score calculated based on duration and frequency of dysfluencies."
                  : "Complete a recording to see your score."}
              </p>
            </div>

            {/* AI Insight */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 h-full">
              <div className="flex items-start gap-3">
                <div className="bg-indigo-100 p-2 rounded-lg">
                  <Activity className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                  <h4 className="font-bold text-slate-700 text-sm mb-1">
                    Therapist Feedback
                  </h4>
                  <ul className="text-sm text-slate-600 space-y-2 mt-2">
                    {results ? (
                      results.feedback.map((tip, idx) => (
                        <li key={idx} className="flex gap-2">
                          <span className="text-indigo-500">•</span> {tip}
                        </li>
                      ))
                    ) : (
                      <li className="italic text-slate-400">
                        Feedback will appear here after analysis.
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpeechTherapistApp;
