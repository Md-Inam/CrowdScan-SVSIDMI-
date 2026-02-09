# 📂 Project Structure

Complete overview of the Enterprise Missing Person Detection System project structure.

## 🏗️ Directory Tree

```
missing-person-detection/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version specification
├── packages.txt               # System dependencies for Streamlit Cloud
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
│
├── README.md                  # Main documentation
├── QUICK_START.md            # Beginner's guide
├── DEPLOYMENT_GUIDE.md       # Deployment instructions
├── CONTRIBUTING.md           # Contribution guidelines
├── PROJECT_STRUCTURE.md      # This file
│
├── Procfile                   # Railway/Heroku process file
├── railway.json              # Railway configuration
├── render.yaml               # Render configuration
│
├── Dockerfile                 # Docker container definition
├── docker-compose.yml        # Docker Compose configuration
│
├── tests/                     # Test files (to be added)
│   ├── __init__.py
│   ├── test_face_detection.py
│   └── test_video_processing.py
│
├── data/                      # Data directory (gitignored)
│   ├── reference_images/
│   └── videos/
│
├── models/                    # Model cache (gitignored)
│   ├── facenet/
│   └── yolo/
│
└── docs/                      # Additional documentation
    ├── api.md
    └── architecture.md
```

## 📄 File Descriptions

### Core Application Files

#### `app.py`
**Purpose**: Main Streamlit application  
**Size**: ~1000 lines  
**Key Components**:
- Page configuration
- Session state management
- Model loading with caching
- Face processing functions
- Video processing pipeline
- UI components (sidebar, results)
- Export functionality

**Key Functions**:
```python
load_models()                    # Load AI models
get_face_embedding()            # Single face embedding
get_face_embedding_batch()      # Batch face processing
process_single_video()          # Video processing pipeline
detect_motion()                 # Motion detection
cosine_similarity()             # Similarity calculation
draw_box()                      # Visual annotations
```

### Configuration Files

#### `requirements.txt`
**Purpose**: Python package dependencies  
**Key Packages**:
- streamlit (Web framework)
- opencv-python-headless (Video processing)
- torch (Deep learning)
- facenet-pytorch (Face recognition)
- plotly (Visualizations)
- ultralytics (YOLO, optional)

#### `.streamlit/config.toml`
**Purpose**: Streamlit app configuration  
**Settings**:
- Theme colors
- Server settings
- Upload limits
- CORS configuration

#### `runtime.txt`
**Purpose**: Specify Python version  
**Content**: `python-3.9.18`

#### `packages.txt`
**Purpose**: System dependencies for Streamlit Cloud  
**Contains**: OpenCV dependencies, system libraries

### Deployment Files

#### `Procfile`
**Purpose**: Process definition for Railway/Heroku  
**Command**: Starts Streamlit server on specified port

#### `railway.json`
**Purpose**: Railway platform configuration  
**Defines**: Build and deployment commands

#### `render.yaml`
**Purpose**: Render platform configuration  
**Defines**: Service type, build, and start commands

#### `Dockerfile`
**Purpose**: Container image definition  
**Features**:
- Python 3.9 slim base
- System dependencies
- Python packages
- Health check
- Exposed port 8501

#### `docker-compose.yml`
**Purpose**: Multi-container orchestration  
**Features**:
- Volume mounts
- Environment variables
- GPU support (optional)
- Auto-restart

### Documentation Files

#### `README.md`
**Purpose**: Main project documentation  
**Sections**:
- Features overview
- Quick start guide
- Installation instructions
- Usage guide
- Deployment options
- Performance examples
- Troubleshooting
- API reference
- Contributing
- License

#### `QUICK_START.md`
**Purpose**: Beginner-friendly tutorial  
**Content**:
- 5-minute setup
- First use tutorial
- Pro tips
- Common mistakes
- Troubleshooting
- Performance expectations

#### `DEPLOYMENT_GUIDE.md`
**Purpose**: Comprehensive deployment instructions  
**Platforms Covered**:
- Streamlit Cloud
- Railway
- Render
- AWS EC2
- Google Cloud Platform
- Docker
- Self-hosted

**Includes**:
- Cost estimates
- Resource requirements
- Security best practices
- Monitoring setup

#### `CONTRIBUTING.md`
**Purpose**: Contribution guidelines  
**Sections**:
- Code of conduct
- How to contribute
- Development setup
- Code style
- Testing
- PR process
- Recognition

#### `PROJECT_STRUCTURE.md`
**Purpose**: This file - project overview

### Other Files

#### `.gitignore`
**Purpose**: Specify files to ignore in Git  
**Ignores**:
- Python cache
- Virtual environments
- Model files
- Video files
- IDE files
- Logs

#### `LICENSE`
**Purpose**: MIT License  
**Allows**: Commercial use, modification, distribution

## 🎯 Key Directories (Created at Runtime)

### `data/`
**Purpose**: User data storage  
**Contents**:
- `reference_images/`: Uploaded reference photos
- `videos/`: Uploaded or processed videos
- `outputs/`: Generated reports and results

**Status**: Gitignored (not tracked)

### `models/`
**Purpose**: AI model cache  
**Contents**:
- `facenet/`: FaceNet VGGFace2 model
- `yolo/`: YOLOv8 model (if used)
- Downloaded automatically on first run

**Status**: Gitignored (models downloaded at runtime)

### `tests/` (To be implemented)
**Purpose**: Unit and integration tests  
**Planned Tests**:
- Face detection accuracy
- Embedding extraction
- Video processing pipeline
- UI components
- Export functionality

## 🔧 Configuration Hierarchy

```
Priority (Highest to Lowest):
1. Command line arguments
2. Environment variables
3. .streamlit/config.toml
4. Streamlit defaults
```

### Environment Variables

```bash
# Optional configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

# Model paths (optional)
FACENET_MODEL_PATH=/path/to/model
YOLO_MODEL_PATH=/path/to/yolo
```

## 📊 Data Flow

```
┌─────────────────┐
│ Reference Image │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Face Detection │ (MTCNN)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Embedding    │ (FaceNet)
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Video Files    │─────▶│ Frame Extract│
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │Motion Detect │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │Face Detection│
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │Batch Encoding│
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Matching   │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   Results    │
                         └──────────────┘
```

## 🚀 Build Process

### Local Development
```bash
1. Clone repository
2. Create virtual environment
3. Install dependencies (pip install -r requirements.txt)
4. Run (streamlit run app.py)
```

### Streamlit Cloud
```bash
1. Connect GitHub repository
2. Auto-detect app.py
3. Install from requirements.txt + packages.txt
4. Deploy
```

### Railway/Render
```bash
1. Detect Procfile or railway.json/render.yaml
2. Run buildCommand (pip install)
3. Run startCommand (streamlit run)
```

### Docker
```bash
1. Build image (docker build -t missing-person .)
2. Run container (docker run -p 8501:8501 missing-person)
Or: docker-compose up
```

## 📦 Deployment Artifacts

### Streamlit Cloud
- No build artifacts
- Dependencies installed at build time
- Models downloaded at runtime

### Railway/Render
- Python packages cached
- Models cached (if persistence enabled)

### Docker
- Container image (~2GB)
- Includes all dependencies
- Models downloaded on first run

### Self-Hosted
- Virtual environment
- Installed packages
- Cached models

## 🔍 Code Organization

### Separation of Concerns

**UI Layer** (Streamlit components):
- Page configuration
- Sidebar
- Main content area
- Progress indicators
- Results display

**Business Logic**:
- Face detection
- Face recognition
- Similarity matching
- Motion detection

**Data Processing**:
- Video frame extraction
- Image preprocessing
- Batch processing
- Parallel execution

**Utilities**:
- Model loading
- File handling
- Export functions

## 🎨 Styling

Styles defined in:
1. Markdown CSS blocks in `app.py`
2. `.streamlit/config.toml` theme settings

Custom classes:
- `.main-header`: Gradient header
- `.metric-card`: Metric display cards
- `.warning-box`: Warning messages
- `.success-box`: Success messages

## 🔒 Security Considerations

### Protected Files
- `.streamlit/secrets.toml` (if created) - Never commit!
- Environment variables - Use platform secrets

### User Data
- Uploaded images - Temporary, cleared on restart
- Video files - Processed but not stored permanently
- Results - Exported as downloads

### Dependencies
- All from PyPI
- Versions pinned in requirements.txt
- Regular security updates recommended

## 📈 Performance Optimizations

### Code Level
- GPU acceleration (FP16)
- Batch processing
- Caching with `@st.cache_resource`
- Parallel video processing

### Deployment Level
- Proper server sizing
- GPU instances (optional)
- CDN for static assets (if applicable)

## 🧪 Testing Strategy

### Current
- Manual testing
- Visual inspection

### Planned
- Unit tests for core functions
- Integration tests for pipeline
- Performance benchmarks
- UI tests

## 📚 Additional Resources

### Not Included but Recommended

Create these as project grows:
- `CHANGELOG.md`: Version history
- `SECURITY.md`: Security policy
- `.github/workflows/`: CI/CD pipelines
- `docs/api.md`: API documentation
- `docs/architecture.md`: System architecture
- `examples/`: Example videos and images

## 🎯 File Size Guidelines

- `app.py`: Keep under 1500 lines
  - Consider splitting into modules if larger
- Documentation: As detailed as needed
- Models: Downloaded automatically (not in repo)
- Videos: Never commit to repo

## 🔄 Update Workflow

When updating:
1. Update code in `app.py`
2. Update dependencies if needed
3. Update relevant documentation
4. Update version in code comments
5. Test locally
6. Test deployment
7. Commit and push
8. Auto-deploy (Streamlit Cloud/Railway)

---

**This structure is optimized for:**
- Easy deployment across multiple platforms
- Clear separation of concerns
- Maintainability
- Scalability
- Contributor onboarding

**Questions?** See README.md or open an issue.
