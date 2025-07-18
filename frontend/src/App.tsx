import React, { useRef, useState, useEffect } from 'react';
import VideoPlayer from './components/VideoPlayer';
import { videoUrl } from './consts';

export interface FaceDetection {
    id: string;
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    label?: string;
  }

interface ProcessingStatus {
  status: string;
  progress: number;
  error?: string;
  stream_url?: string;
  download_url?: string;
  file_exists?: boolean;
  file_size?: number;
  filter_type?: string;
  start_time?: number;
  end_time?: number;
}

interface FilterOption {
  key: string;
  name: string;
  description: string;
  preview?: string;
}

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const processedVideoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [response, setResponse] = useState<string>('');
  const [jobId, setJobId] = useState<string>('');
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  
  // New features state
  const [availableFilters, setAvailableFilters] = useState<{[key: string]: string}>({});
  const [selectedFilter, setSelectedFilter] = useState<string>('grayscale');
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string>(videoUrl);
  const [uploadedVideo, setUploadedVideo] = useState<any>(null);
  const [startTime, setStartTime] = useState<number | undefined>(undefined);
  const [endTime, setEndTime] = useState<number | undefined>(undefined);
  const [videoDuration, setVideoDuration] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [uiMode, setUiMode] = useState<'standard' | 'theater' | 'editor'>('standard');

  // New state for technical transparency
  const [showTechnicalInfo, setShowTechnicalInfo] = useState<boolean>(false);
  const [technicalInfo, setTechnicalInfo] = useState<any>(null);
  const [processingLogs, setProcessingLogs] = useState<any>(null);
  const [modelsInfo, setModelsInfo] = useState<any>(null);

  // Load available filters on component mount
  useEffect(() => {
    fetchAvailableFilters();
    fetchTechnicalInfo();
  }, []);

  const fetchAvailableFilters = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8080/available-filters');
      const data = await res.json();
      setAvailableFilters(data.filters || {});
      console.log('Available filters:', data.filters);
      
      // Ensure selectedFilter is valid, set to first available filter if not
      if (data.filters && !data.filters[selectedFilter]) {
        const firstFilterKey = Object.keys(data.filters)[0];
        if (firstFilterKey) {
          setSelectedFilter(firstFilterKey);
          console.log('Set selected filter to:', firstFilterKey);
        }
      } else {
        console.log('Current selected filter:', selectedFilter, 'is valid');
      }
    } catch (error) {
      console.error('Error fetching filters:', error);
    }
  };

  const fetchTechnicalInfo = async () => {
    try {
      const [techRes, modelsRes] = await Promise.all([
        fetch('http://127.0.0.1:8080/technical-info'),
        fetch('http://127.0.0.1:8080/models-info')
      ]);
      
      const techData = await techRes.json();
      const modelsData = await modelsRes.json();
      
      setTechnicalInfo(techData);
      setModelsInfo(modelsData);
      console.log('Technical info loaded:', techData);
    } catch (error) {
      console.error('Error fetching technical info:', error);
    }
  };

  const fetchProcessingLogs = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8080/processing-logs');
      const data = await res.json();
      setProcessingLogs(data);
      console.log('Processing logs:', data);
    } catch (error) {
      console.error('Error fetching processing logs:', error);
    }
  };

  const pingBackend = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8080/hello-world');
      const data = await res.text();
      setResponse(data);
      console.log('Backend response:', data);
    } catch (error) {
      console.error('Error pinging backend:', error);
      setResponse('Error connecting to backend');
    }
  };

  const uploadVideo = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('video', file);
      
      const res = await fetch('http://127.0.0.1:8080/upload-video', {
        method: 'POST',
        body: formData
      });
      
      const data = await res.json();
      
      if (data.video_url) {
        setUploadedVideo(data);
        setCurrentVideoUrl(`http://127.0.0.1:8080${data.video_url}`);
        console.log('Video uploaded:', data);
      } else {
        throw new Error(data.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Error uploading video:', error);
      alert('Failed to upload video');
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      uploadVideo(file);
    }
  };

  const processVideo = async () => {
    try {
      setIsProcessing(true);
      setProcessingStatus(null);
      setProcessedVideoUrl('');
      
      console.log('=== PROCESSING DEBUG INFO ===');
      console.log('Selected filter:', selectedFilter);
      console.log('Available filters:', availableFilters);
      console.log('Start time:', startTime);
      console.log('End time:', endTime);
      console.log('Video duration:', videoDuration);
      console.log('Current video URL:', currentVideoUrl);
      
      const requestData: any = {
        video_url: currentVideoUrl,
        filter_type: selectedFilter,
        playback_speed: playbackSpeed
      };
      
      // Add timeframe if specified
      if (startTime !== undefined && endTime !== undefined) {
        requestData.start_time = startTime;
        requestData.end_time = endTime;
        console.log('Adding timeframe to request:', { start_time: startTime, end_time: endTime });
      } else {
        console.log('No timeframe specified - will apply filter to entire video');
      }
      
      console.log('Playback speed:', playbackSpeed);
      
      console.log('Request data being sent to backend:', requestData);
      
      const res = await fetch('http://127.0.0.1:8080/process-video', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      const data = await res.json();
      console.log('Backend response:', data);
      
      if (data.job_id) {
        setJobId(data.job_id);
        console.log('Starting to poll job:', data.job_id);
        pollProcessingStatus(data.job_id);
      } else {
        throw new Error(data.error || 'No job_id received');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setProcessingStatus({
        status: 'error',
        progress: 0,
        error: 'Failed to start video processing'
      });
      setIsProcessing(false);
    }
  };

  const pollProcessingStatus = async (jobId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8080/processing-status/${jobId}`);
        const status: ProcessingStatus = await res.json();
        
        console.log('Status update:', status);
        setProcessingStatus(status);
        
        if (status.status === 'completed') {
          clearInterval(pollInterval);
          setIsProcessing(false);
          
          if (status.stream_url) {
            const fullStreamUrl = `http://127.0.0.1:8080${status.stream_url}`;
            console.log('Setting processed video URL:', fullStreamUrl);
            setProcessedVideoUrl(fullStreamUrl);
          } else {
            console.error('No stream_url in completed status');
          }
        } else if (status.status === 'error') {
          clearInterval(pollInterval);
          setIsProcessing(false);
          console.error('Processing failed:', status.error);
        }
      } catch (error) {
        console.error('Error polling status:', error);
        clearInterval(pollInterval);
        setIsProcessing(false);
      }
    }, 1000);
  };



  const onVideoMetadataLoaded = () => {
    if (videoRef.current) {
      setVideoDuration(videoRef.current.duration);
    }
  };

  const getStatusMessage = () => {
    if (!processingStatus) return '';
    
    switch (processingStatus.status) {
      case 'downloading':
        return 'Downloading video...';
      case 'processing':
        return `Processing with ${availableFilters[selectedFilter] || selectedFilter}... ${processingStatus.progress}%`;
      case 'completed':
        return `Processing completed with ${availableFilters[selectedFilter] || selectedFilter}!`;
      case 'error':
        return `Error: ${processingStatus.error}`;
      default:
        return processingStatus.status;
    }
  };

  const getStatusColor = () => {
    if (!processingStatus) return '#6c757d';
    
    switch (processingStatus.status) {
      case 'completed':
        return '#28a745';
      case 'error':
        return '#dc3545';
      case 'processing':
      case 'downloading':
        return '#007bff';
      default:
        return '#6c757d';
    }
  };

  const renderFilterGallery = () => {
    console.log('Rendering filter gallery. Current selectedFilter:', selectedFilter);
    console.log('Available filters:', availableFilters);
    
    return (
      <div className="filter-gallery">
        <h3>🎨 Effect Gallery</h3>
        <div className="filter-grid">
          {Object.entries(availableFilters).map(([key, name]) => {
            const isSelected = selectedFilter === key;
            console.log(`Filter ${key}: selected=${isSelected}`);
            return (
              <div
                key={key}
                className={`filter-option ${isSelected ? 'selected' : ''}`}
                onClick={() => {
                  setSelectedFilter(key);
                  console.log('Selected filter:', key, name);
                }}
              >
                {name}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderTimeframeControls = () => {
    return (
      <div className="timeframe-controls">
        <h3>⏱️ Timeframe Selection</h3>
        <div className="time-inputs">
          <div className="time-input-group">
            <label>Start Time (seconds):</label>
            <input
              type="number"
              min="0"
              max={videoDuration}
              step="0.1"
              value={startTime || ''}
              onChange={(e) => {
                const value = e.target.value ? parseFloat(e.target.value) : undefined;
                console.log('Start time changed to:', value);
                setStartTime(value);
              }}
              placeholder="Start time"
            />
          </div>
          <div className="time-input-group">
            <label>End Time (seconds):</label>
            <input
              type="number"
              min="0"
              max={videoDuration}
              step="0.1"
              value={endTime || ''}
              onChange={(e) => {
                const value = e.target.value ? parseFloat(e.target.value) : undefined;
                console.log('End time changed to:', value);
                setEndTime(value);
              }}
              placeholder="End time"
            />
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => {
              console.log('Clearing timeframe values');
              setStartTime(undefined);
              setEndTime(undefined);
            }}
          >
            Clear
          </button>
        </div>
        {videoDuration > 0 && (
          <p className="duration-info">Video duration: {videoDuration.toFixed(1)}s</p>
        )}
        {startTime !== undefined && endTime !== undefined && (
          <p className="timeframe-info">
            Filter will apply from {startTime}s to {endTime}s ({(endTime - startTime).toFixed(1)}s duration)
          </p>
        )}
      </div>
    );
  };

  const renderSpeedControls = () => {
    const speedOptions = [
      { value: 0.25, label: '0.25x (Slow Motion)' },
      { value: 0.5, label: '0.5x (Half Speed)' },
      { value: 1.0, label: '1x (Normal)' },
      { value: 2.0, label: '2x (Fast)' },
      { value: 4.0, label: '4x (Time-lapse)' }
    ];

    return (
      <div className="speed-controls">
        <h3>🚀 Playback Speed</h3>
        <div className="speed-options">
          {speedOptions.map((option) => (
            <button
              key={option.value}
              className={`speed-option ${playbackSpeed === option.value ? 'selected' : ''}`}
              onClick={() => {
                console.log('Speed changed to:', option.value);
                setPlaybackSpeed(option.value);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
        <p className="speed-info">
          Current speed: {playbackSpeed}x 
          {playbackSpeed < 1 && ' (Slow Motion)'}
          {playbackSpeed > 1 && ' (Time-lapse)'}
        </p>
      </div>
    );
  };

  const renderUIControls = () => {
    return (
      <div className="ui-controls">
        <h3>View Mode</h3>
        <div className="mode-buttons">
          <button
            className={`btn ${uiMode === 'standard' ? 'active' : ''}`}
            onClick={() => setUiMode('standard')}
            data-mode="standard"
          >
            Standard
          </button>
          <button
            className={`btn ${uiMode === 'theater' ? 'active' : ''}`}
            onClick={() => setUiMode('theater')}
            data-mode="theater"
          >
            Theater
          </button>
          <button
            className={`btn ${uiMode === 'editor' ? 'active' : ''}`}
            onClick={() => setUiMode('editor')}
            data-mode="editor"
          >
            Editor
          </button>
        </div>
      </div>
    );
  };

  const renderVideoSection = () => {
    return (
      <div className="video-section">
        <div className="video-card">
          <h2>Original Video</h2>
          <div className="video-container">
            <VideoPlayer
              ref={videoRef}
              src={currentVideoUrl}
              onLoadedMetadata={onVideoMetadataLoaded}
            />
          </div>
          <div className="video-info">
            {uploadedVideo ? (
              <p>{uploadedVideo.filename} ({(uploadedVideo.size / 1024 / 1024).toFixed(2)} MB)</p>
            ) : (
              <p>Default demo video</p>
            )}
          </div>
        </div>
        
        <div className="video-card">
          <h2>Processed Video</h2>
          <div className="video-player-container">
            {processedVideoUrl ? (
              <VideoPlayer
                ref={processedVideoRef}
                src={processedVideoUrl}
                onLoadedMetadata={() => console.log('Processed video loaded')}
              />
            ) : (
              <div className="video-placeholder">
                <div className="placeholder-content">
                  <h3>Processed video will appear here</h3>
                  <p>Select a filter and click "Process Video" to begin</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderTechnicalTransparency = () => {
    if (!showTechnicalInfo) return null;

    return (
      <div className="technical-section">
        <div className="technical-panel">
          <h2>Technical Transparency</h2>
          <p>Proving we use both Haar Cascade AND MediaPipe as required</p>
          
          <div className="tech-tabs">
            <div className="tech-content">
              {/* Assignment Compliance */}
              <div className="compliance-section">
                <h3>Assignment Compliance</h3>
                {technicalInfo?.assignment_compliance && (
                  <div className="compliance-grid">
                    <div className="compliance-item">
                      <h4>Requirement Met</h4>
                      <p>{technicalInfo.assignment_compliance.requirement}</p>
                      <p><strong>Status:</strong> {technicalInfo.assignment_compliance.implementation}</p>
                    </div>
                    
                    <div className="compliance-item">
                      <h4>Haar Cascade Models</h4>
                      <ul>
                        {technicalInfo.assignment_compliance.models_used.map((model: string, i: number) => (
                          <li key={i}><code>{model}</code></li>
                        ))}
                      </ul>
                      <p><strong>Method:</strong> <code>{technicalInfo.assignment_compliance.detection_method}</code></p>
                    </div>
                    
                    <div className="compliance-item">
                      <h4>Enhancement</h4>
                      <p><strong>Technology:</strong> {technicalInfo.enhancement?.technology}</p>
                      <p><strong>Purpose:</strong> {technicalInfo.enhancement?.purpose}</p>
                      <p><strong>Reason:</strong> {technicalInfo.enhancement?.reason}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Processing Pipeline */}
              <div className="pipeline-section">
                <h3>Hybrid Processing Pipeline</h3>
                {technicalInfo?.hybrid_fusion && (
                  <div className="pipeline-steps">
                    <div className="step">
                      <span className="step-icon">1</span>
                      <div>
                        <strong>Step 1: Haar Cascade Detection</strong>
                        <p>Uses <code>cv2.CascadeClassifier.detectMultiScale()</code> with pre-trained models</p>
                      </div>
                    </div>
                    <div className="step">
                      <span className="step-icon">2</span>
                      <div>
                        <strong>Step 2: MediaPipe Segmentation</strong>
                        <p>AI-powered pixel-perfect person/background separation</p>
                      </div>
                    </div>
                    <div className="step">
                      <span className="step-icon">3</span>
                      <div>
                        <strong>Step 3: Hybrid Fusion</strong>
                        <p>Combines Haar detection with MediaPipe for maximum accuracy</p>
                      </div>
                    </div>
                    <div className="step">
                      <span className="step-icon">4</span>
                      <div>
                        <strong>Step 4: Filter Application</strong>
                        <p>Applies selected visual effects using OpenCV</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Model Details */}
              <div className="models-section">
                <h3>Model Specifications</h3>
                {modelsInfo && (
                  <div className="models-grid">
                    <div className="model-card">
                      <h4>OpenCV Haar Cascade</h4>
                      <p><strong>Version:</strong> {modelsInfo.opencv_version}</p>
                      <div className="model-details">
                        <h5>Face Detection Model:</h5>
                        <p><code>{modelsInfo.haar_cascade_models?.face_model?.file}</code></p>
                        <p><strong>Purpose:</strong> {modelsInfo.haar_cascade_models?.face_model?.purpose}</p>
                        
                        <h5>Body Detection Model:</h5>
                        <p><code>{modelsInfo.haar_cascade_models?.body_model?.file}</code></p>
                        <p><strong>Purpose:</strong> {modelsInfo.haar_cascade_models?.body_model?.purpose}</p>
                      </div>
                    </div>

                    <div className="model-card">
                      <h4>Google MediaPipe</h4>
                      <p><strong>Model:</strong> {modelsInfo.mediapipe_model?.name}</p>
                      <p><strong>Version:</strong> {modelsInfo.mediapipe_model?.version}</p>
                      <p><strong>Purpose:</strong> {modelsInfo.mediapipe_model?.purpose}</p>
                      <p><strong>Input:</strong> {modelsInfo.mediapipe_model?.input}</p>
                      <p><strong>Output:</strong> {modelsInfo.mediapipe_model?.output}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Live Processing Logs */}
              <div className="logs-section">
                <h3>Real-Time Processing Logs</h3>
                <button 
                  className="btn btn-secondary"
                  onClick={fetchProcessingLogs}
                >
                  Refresh Processing Logs
                </button>
                
                {processingLogs && (
                  <div className="logs-display">
                    <p><strong>Recent Frames Processed:</strong> {processingLogs.log_count}</p>
                    
                    {processingLogs.recent_logs?.slice(-3).map((log: any, i: number) => (
                      <div key={i} className="log-entry">
                        <h4>Frame at {log.timestamp?.toFixed(1)}s</h4>
                        <div className="log-grid">
                          <div className="log-step">
                            <h5>Haar Cascade</h5>
                            <p><strong>Method:</strong> {log.step1_haar?.method}</p>
                            <p><strong>Detection:</strong> {log.step1_haar?.detection_found ? `✅ ${log.step1_haar.detection_type}` : '❌ None'}</p>
                            <p><strong>Time:</strong> {log.step1_haar?.processing_time_ms?.toFixed(1)}ms</p>
                          </div>
                          
                          <div className="log-step">
                            <h5>MediaPipe</h5>
                            <p><strong>Method:</strong> {log.step2_mediapipe?.method}</p>
                            <p><strong>Model:</strong> {log.step2_mediapipe?.model}</p>
                            <p><strong>Time:</strong> {log.step2_mediapipe?.processing_time_ms?.toFixed(1)}ms</p>
                          </div>
                          
                          <div className="log-step">
                            <h5>Fusion</h5>
                            <p><strong>Method:</strong> {log.step3_fusion?.method}</p>
                            <p><strong>Threshold:</strong> {log.step3_fusion?.threshold_used}</p>
                            <p><strong>Haar Influenced:</strong> {log.step3_fusion?.haar_influenced ? '✅ Yes' : '❌ No'}</p>
                          </div>
                        </div>
                        
                        <div className="log-summary">
                          <p><strong>Filter Applied:</strong> {log.final_result?.filter_applied}</p>
                          <p><strong>Total Time:</strong> {log.final_result?.total_processing_time_ms?.toFixed(1)}ms</p>
                          <p><strong>Approach:</strong> {log.final_result?.hybrid_approach}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`app-container ${uiMode}`}>
      <header className="header">
        <h1>AI Video Studio</h1>
        <p>Professional video editing with AI-powered person detection</p>
      </header>

      <main className="main">
        {renderVideoSection()}
        
        <div className="controls-section">
          <div className="control-panel">
            <div className="upload-section">
              <h3>Video Source</h3>
              <div className="upload-controls">
                <button
                  className="btn btn-upload"
                  onClick={() => fileInputRef.current?.click()}
                >
                  Upload Your Video
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setCurrentVideoUrl(videoUrl);
                    setUploadedVideo(null);
                  }}
                >
                  Use Demo Video
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
              </div>
            </div>

            {renderFilterGallery()}
            {renderTimeframeControls()}
            {renderSpeedControls()}
            {renderUIControls()}

            <div className="action-section">
              <h3>Actions</h3>
              <div className="action-buttons">
                <button
                  className="btn btn-process"
                  onClick={processVideo}
                  disabled={isProcessing}
                >
                  {isProcessing ? 'Processing...' : 'Process Video'}
                </button>
                
                <button
                  className="btn btn-secondary"
                  onClick={pingBackend}
                >
                  Test Connection
                </button>

                <button
                  className="btn btn-info"
                  onClick={() => setShowTechnicalInfo(!showTechnicalInfo)}
                >
                  {showTechnicalInfo ? 'Hide Tech Details' : 'Show Tech Details'}
                </button>

                {processedVideoUrl && (
                  <a
                    href={`http://127.0.0.1:8080/download-processed/${jobId}`}
                    className="btn btn-download"
                    download
                  >
                    Download Result
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>

        {renderTechnicalTransparency()}

        {processingStatus && (
          <div className="status-section">
            <div 
              className="status-card"
              style={{ borderColor: getStatusColor() }}
            >
              <h3 style={{ color: getStatusColor() }}>
                {getStatusMessage()}
              </h3>
              
              {processingStatus.status === 'processing' && (
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ 
                      width: `${processingStatus.progress}%`,
                      backgroundColor: getStatusColor()
                    }}
                  />
                </div>
              )}
              
              {processingStatus.status === 'completed' && (
                <div className="completion-info">
                  <p>✅ File exists: {processingStatus.file_exists ? 'Yes' : 'No'}</p>
                  {processingStatus.file_size && (
                    <p>📊 File size: {(processingStatus.file_size / 1024 / 1024).toFixed(2)} MB</p>
                  )}
                  <p>🎨 Filter: {availableFilters[selectedFilter] || selectedFilter}</p>
                  {startTime !== undefined && endTime !== undefined && (
                    <p>⏱️ Timeframe: {startTime}s - {endTime}s</p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </main>


    </div>
  );
};

export default App; 