# 🚀 Deployment Guide

Complete guide to deploying the Enterprise Missing Person Detection System on various platforms.

## 📋 Prerequisites

Before deploying, ensure you have:
- Git installed
- GitHub account (for most deployment methods)
- Basic command line knowledge

## 🎯 Quick Deployment Options

| Platform | Best For | GPU Support | Free Tier |
|----------|----------|-------------|-----------|
| **Streamlit Cloud** | Quick demos, prototypes | ❌ No | ✅ Yes (Limited) |
| **Railway** | Production, large datasets | ✅ Yes (Paid) | ✅ Yes (Limited) |
| **Render** | Medium workloads | ❌ No | ✅ Yes (Limited) |
| **AWS EC2** | Enterprise, full control | ✅ Yes | ❌ No |
| **Google Cloud** | Enterprise, ML workloads | ✅ Yes | ✅ Yes ($300 credit) |

## 1️⃣ Streamlit Cloud (Easiest)

### Pros
- Free tier available
- Zero configuration
- Automatic HTTPS
- Easy updates via GitHub

### Cons
- Limited resources (1GB RAM)
- No GPU support
- Slow for large videos
- Public by default

### Steps

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/missing-person-detection.git
git push -u origin main
```

2. **Deploy**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Click "New app"
- Select your repository
- Choose `app.py` as main file
- Click "Deploy"

3. **Configure (Optional)**
- Set environment variables in Advanced settings
- Adjust Python version in `runtime.txt`

### Limitations
⚠️ Streamlit Cloud is best for demos. For production use with videos, use Railway or dedicated server.

## 2️⃣ Railway (Recommended for Production)

### Pros
- GPU support available
- $5 free credit/month
- Automatic HTTPS
- Easy scaling
- Great performance

### Cons
- Paid after free tier
- Limited free resources

### Steps

#### Method A: GitHub Deploy (Easiest)

1. **Push to GitHub** (if not already done)

2. **Create Railway Project**
- Go to [railway.app](https://railway.app)
- Sign up with GitHub
- Click "New Project"
- Select "Deploy from GitHub repo"
- Choose your repository
- Railway auto-detects Streamlit app

3. **Configure**
- Railway automatically reads `Procfile` and `railway.json`
- App deploys automatically

4. **Get URL**
- Click on deployment
- Copy the provided URL

#### Method B: Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize
railway init

# Deploy
railway up

# Get URL
railway open
```

### GPU Configuration

To enable GPU (paid plan required):

1. Go to project settings
2. Select "Service Variables"
3. Add: `RAILWAY_GPU=nvidia-tesla-t4`
4. Redeploy

## 3️⃣ Render

### Pros
- Free tier available
- Automatic HTTPS
- Good documentation

### Cons
- Slower than Railway
- No GPU support
- Limited free tier

### Steps

1. **Create `render.yaml`** (already included)

2. **Push to GitHub**

3. **Deploy**
- Go to [render.com](https://render.com)
- Click "New Web Service"
- Connect your repository
- Render auto-detects and deploys

## 4️⃣ AWS EC2 (Advanced)

### Pros
- Full control
- GPU instances available
- Highly scalable
- Production-ready

### Cons
- Requires AWS knowledge
- Manual setup
- Costs can add up

### Steps

1. **Launch EC2 Instance**
```bash
# Recommended: Ubuntu 22.04 LTS
# Instance type: 
#   - CPU only: t3.xlarge (4 vCPU, 16GB RAM)
#   - GPU: g4dn.xlarge (NVIDIA T4)
```

2. **SSH into Instance**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install system dependencies
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

4. **Setup Application**
```bash
# Clone repository
git clone https://github.com/yourusername/missing-person-detection.git
cd missing-person-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

5. **Run with Systemd**

Create `/etc/systemd/system/missing-person.service`:

```ini
[Unit]
Description=Missing Person Detection System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/missing-person-detection
Environment="PATH=/home/ubuntu/missing-person-detection/venv/bin"
ExecStart=/home/ubuntu/missing-person-detection/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable missing-person
sudo systemctl start missing-person
```

6. **Setup Nginx Reverse Proxy**

```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/missing-person`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/missing-person /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

7. **Setup SSL (Optional but Recommended)**

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

### GPU Setup (for g4dn instances)

```bash
# Install NVIDIA drivers
sudo apt install nvidia-driver-525 -y

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda -y

# Verify
nvidia-smi
```

## 5️⃣ Google Cloud Platform

### Pros
- $300 free credit
- Excellent GPU support
- Global infrastructure
- Auto-scaling

### Cons
- Complex setup
- Can be expensive

### Steps

1. **Create Compute Engine Instance**
```bash
gcloud compute instances create missing-person-detector \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --maintenance-policy=TERMINATE
```

2. **SSH and Setup** (similar to AWS EC2 steps above)

3. **Install NVIDIA Drivers**
```bash
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
```

## 🔧 Environment Variables

For production deployments, set these environment variables:

```bash
# Optional: Custom model paths
FACENET_MODEL_PATH=/path/to/model
YOLO_MODEL_PATH=/path/to/yolo

# Optional: Performance tuning
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
STREAMLIT_SERVER_ENABLE_CORS=false
```

## 📊 Resource Requirements

### Minimum (CPU Only)
- 4 CPU cores
- 8GB RAM
- 20GB storage
- Can process: ~5 videos/hour

### Recommended (CPU)
- 8 CPU cores
- 16GB RAM
- 50GB storage
- Can process: ~20 videos/hour

### Optimal (GPU)
- 8 CPU cores
- 16GB RAM
- NVIDIA T4 or better
- 100GB storage
- Can process: ~100 videos/hour

## 🐛 Common Issues

### Issue: Out of Memory

**Solution 1**: Reduce batch size in `app.py`:
```python
# Line 250
if len(face_batch) >= 8:  # Changed from 16
```

**Solution 2**: Increase server memory

**Solution 3**: Process fewer videos in parallel

### Issue: Models Not Downloading

**Solution**: Pre-download models:
```bash
python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"
```

### Issue: Slow Processing

**Solution 1**: Enable GPU
**Solution 2**: Increase frame sampling rate
**Solution 3**: Enable motion detection

### Issue: Port Already in Use

**Solution**: Change port in command:
```bash
streamlit run app.py --server.port=8502
```

## 🔒 Security Best Practices

1. **Use HTTPS**: Always enable SSL/TLS
2. **Restrict Access**: Use firewall rules
3. **Environment Variables**: Never commit secrets
4. **Update Regularly**: Keep dependencies updated
5. **Monitor Logs**: Watch for suspicious activity

## 📈 Monitoring

### Basic Monitoring (All Platforms)

Check logs:
```bash
# Streamlit Cloud: View in dashboard
# Railway: railway logs
# EC2: journalctl -u missing-person -f
```

### Advanced Monitoring (AWS/GCP)

- CloudWatch (AWS)
- Cloud Monitoring (GCP)
- Custom metrics with Prometheus + Grafana

## 🔄 Updates & Maintenance

### Update Application

**Streamlit Cloud / Railway**:
Just push to GitHub:
```bash
git add .
git commit -m "Update"
git push
```
Auto-deploys!

**EC2 / Self-Hosted**:
```bash
cd missing-person-detection
git pull
sudo systemctl restart missing-person
```

### Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

## 💰 Cost Estimates

### Streamlit Cloud
- Free tier: $0/month (limited resources)
- Not suitable for production

### Railway
- Free: $5 credit/month
- Starter: ~$10-20/month (no GPU)
- Pro with GPU: ~$50-100/month

### AWS EC2
- t3.xlarge: ~$150/month
- g4dn.xlarge (GPU): ~$400/month

### GCP
- Similar to AWS
- Use cost calculator: [cloud.google.com/products/calculator](https://cloud.google.com/products/calculator)

## 🎓 Best Practices

1. **Start Small**: Begin with Streamlit Cloud or Railway free tier
2. **Test Thoroughly**: Test with small datasets first
3. **Monitor Costs**: Set up billing alerts
4. **Optimize**: Use all speed optimizations
5. **Scale Gradually**: Increase resources as needed
6. **Backup Data**: Regular backups of results
7. **Document**: Keep deployment notes

## 📞 Support

- **Deployment Issues**: Open GitHub issue
- **Platform-Specific**: Check platform documentation
- **Emergency**: support@yourproject.com

## 🔗 Useful Links

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Railway Docs](https://docs.railway.app)
- [AWS EC2 Tutorial](https://docs.aws.amazon.com/ec2/)
- [GCP Compute Engine](https://cloud.google.com/compute/docs)

---

**Need help?** Open an issue on GitHub or check our troubleshooting guide.
