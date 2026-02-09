# 🚀 Quick Start Guide

Get the Enterprise Missing Person Detection System running in 5 minutes!

## ⚡ Super Quick Start (Local)

### For Windows

```bash
# 1. Clone repository
git clone https://github.com/yourusername/missing-person-detection.git
cd missing-person-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

### For macOS/Linux

```bash
# 1. Clone repository
git clone https://github.com/yourusername/missing-person-detection.git
cd missing-person-detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 First Use Tutorial

### Step 1: Prepare Reference Image
- Get a clear photo of the missing person
- Face should be visible and front-facing
- Good lighting preferred
- Formats: JPG, PNG

### Step 2: Prepare Test Video(s)
For testing, you can:
- Use your webcam to record a short video
- Download sample CCTV footage
- Use any MP4, AVI, or MOV file

### Step 3: Upload Reference Image
1. Open the app (http://localhost:8501)
2. Look at left sidebar
3. Click "Browse files" under "Reference Image"
4. Select your reference photo
5. Wait for face detection confirmation

### Step 4: Select Videos

**Option A - Upload (Easy for beginners)**
1. Click "Upload Files" radio button
2. Click "Browse files"
3. Select 1-3 test videos (max 200MB each)
4. Wait for upload

**Option B - Local Folder (Better for many videos)**
1. Click "Local Folder Path" radio button
2. Create a folder: `C:\CCTV_Footage` (Windows) or `~/CCTV_Footage` (Mac/Linux)
3. Copy your videos into this folder
4. Enter the folder path in the text box
5. System finds all videos automatically

### Step 5: Configure Settings

**For First Test (Fast Processing):**
- Confidence Threshold: 0.55 (default)
- Min Face Size: 40 (default)
- Frame Sampling Rate: 30 (check every 30th frame)
- Motion Detection: ✅ Enabled
- Parallel Processing: 2

**For Thorough Search:**
- Confidence Threshold: 0.45 (more sensitive)
- Min Face Size: 30 (detect smaller faces)
- Frame Sampling Rate: 5 (check more frames)
- Motion Detection: ❌ Disabled
- Parallel Processing: 1

### Step 6: Start Processing
1. Click "🚀 Start Batch Processing" button
2. Watch real-time progress
3. See detections appear live
4. Wait for completion

### Step 7: Review Results
- Browse through detected matches
- Check confidence scores
- View annotated frames
- Filter results as needed

### Step 8: Export Report
Choose export format:
- **JSON**: For further analysis
- **CSV**: For Excel/spreadsheet
- **TXT**: For human-readable report

## 🎬 Video Tutorial

**Coming Soon**: Step-by-step video tutorial

For now, follow the steps above!

## 💡 Pro Tips

### 🎯 Getting Best Results

1. **Reference Image Quality Matters**
   - Use recent photo if possible
   - Clear, front-facing shot
   - Good resolution (not blurry)
   - Good lighting

2. **Optimize for Your Hardware**
   - **Have GPU?** 
     - Keep all optimizations enabled
     - Use higher parallel processing (4-8)
   - **CPU Only?**
     - Use frame sampling: 20-30
     - Enable motion detection
     - Parallel processing: 2-3

3. **Balance Speed vs Accuracy**
   - **Need it fast?** 
     - Frame sampling: 30
     - Motion detection: ON
     - Confidence: 0.55
   - **Need thoroughness?**
     - Frame sampling: 5
     - Motion detection: OFF
     - Confidence: 0.45

4. **Processing Large Datasets**
   - Process 50 videos at a time
   - Use parallel processing
   - Monitor memory usage
   - Take breaks between batches

### ⚠️ Common Mistakes

1. **❌ Using Blurry Reference Photo**
   - ✅ Use clear, high-quality image

2. **❌ Setting Threshold Too High**
   - ✅ Start with 0.55, adjust down if needed

3. **❌ Sampling Too Aggressively**
   - ✅ If missing detections, reduce sampling rate

4. **❌ Skipping Motion Detection**
   - ✅ Enable it - saves huge time with minimal accuracy loss

5. **❌ Processing Too Many Videos at Once**
   - ✅ Start with 5-10 videos to test settings

## 🐛 Troubleshooting

### Problem: "No face detected in reference image"
**Solutions:**
- Ensure face is clearly visible
- Try different photo
- Check lighting
- Ensure face is front-facing

### Problem: "No matches found"
**Solutions:**
- Lower confidence threshold (try 0.45)
- Reduce frame sampling (try 5-10)
- Disable motion detection
- Try different reference photo
- Check if person appears in videos

### Problem: App is slow
**Solutions:**
- Increase frame sampling (30-60)
- Enable motion detection
- Reduce parallel processing
- Close other applications
- Check CPU/RAM usage

### Problem: Out of memory error
**Solutions:**
- Process fewer videos at once
- Reduce parallel processing to 1
- Reduce batch size in code
- Restart application
- Add more RAM

### Problem: Can't upload videos
**Solutions:**
- Check file size (max 200MB per file)
- Use local folder path instead
- Verify file format (MP4, AVI, MOV)

## 📞 Get Help

**Still stuck?**
1. Check [README.md](README.md) for detailed docs
2. Read [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment
3. Open issue on GitHub
4. Check GitHub Discussions

## 🎓 Next Steps

Once you're comfortable:
1. Try deploying to Streamlit Cloud (free!)
2. Experiment with different settings
3. Process larger datasets
4. Contribute improvements
5. Share with others

## ✅ Checklist

Before your first run:
- [ ] Python 3.9+ installed
- [ ] Git installed (optional)
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Reference image ready
- [ ] Test video(s) ready
- [ ] 8GB+ RAM available

## 🌟 Success Story Template

After successful detection, share your experience:
```
Found matches in X videos!
Processing time: Y minutes
Hardware: [CPU/GPU specs]
Settings used: [Your settings]
Tips: [What worked well]
```

## 📊 Expected Performance

**On Typical Laptop (8GB RAM, CPU only):**
- 1 hour video: ~5 minutes
- 10 videos (1 hour each): ~30 minutes
- Settings: Frame sampling 30, Motion detection ON

**On Gaming PC (16GB RAM, NVIDIA GPU):**
- 1 hour video: ~1 minute
- 10 videos (1 hour each): ~5 minutes
- Settings: Frame sampling 15, Motion detection ON, GPU enabled

## 🎉 You're Ready!

That's it! You're now ready to use the system.

**Remember**: This tool helps find missing persons. Use it responsibly and ethically.

---

**Questions?** Open an issue or check the full README.md

**Good luck! 🍀**
