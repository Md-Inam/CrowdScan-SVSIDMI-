# Contributing to Enterprise Missing Person Detection System

First off, thank you for considering contributing to this project! 🎉

## 🤝 Code of Conduct

Be respectful, inclusive, and professional. This project aims to help find missing persons - keep that mission in mind.

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

**Before submitting a bug report:**
- Check existing issues to avoid duplicates
- Collect information about the bug:
  - Steps to reproduce
  - Expected vs actual behavior
  - Screenshots/videos if applicable
  - System information (OS, Python version, GPU)

**Submit bug report:**
1. Use GitHub Issues
2. Use the bug report template
3. Include all relevant information
4. Be clear and descriptive

### 💡 Suggesting Enhancements

**Before submitting:**
- Check if enhancement already exists
- Verify it aligns with project goals

**Submit enhancement:**
1. Use GitHub Issues
2. Use the feature request template
3. Explain the use case
4. Provide examples if possible

### 🔧 Pull Requests

**Process:**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/AmazingFeature`
7. Open Pull Request

**PR Guidelines:**
- One feature per PR
- Update documentation
- Add tests if applicable
- Follow code style
- Keep commits atomic

## 🏗️ Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/missing-person-detection.git
cd missing-person-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

### 4. Run Tests
```bash
pytest tests/  # If tests exist
```

### 5. Run Application
```bash
streamlit run app.py
```

## 📝 Code Style

### Python
- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for functions
- Keep functions focused and small
- Comment complex logic

**Example:**
```python
def get_face_embedding(img: Image.Image, resnet: torch.nn.Module, 
                       device: torch.device) -> Optional[np.ndarray]:
    """
    Extract facial embedding from image.
    
    Args:
        img: PIL Image containing face
        resnet: FaceNet model
        device: PyTorch device (cuda/cpu)
    
    Returns:
        512-dimensional embedding or None if extraction fails
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Failed to extract embedding: {e}")
        return None
```

### Streamlit
- Keep UI code separate from logic
- Use caching appropriately
- Optimize for performance
- Provide helpful user messages

### Git Commits
- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Reference issues: "Fix #123: Resolve memory leak"

**Good commit messages:**
```
Add batch processing for face embeddings
Fix memory leak in video processing
Update README with deployment instructions
Optimize motion detection algorithm
```

## 🧪 Testing

### Manual Testing Checklist
- [ ] Upload reference image works
- [ ] Video file selection works
- [ ] Processing completes without errors
- [ ] Results display correctly
- [ ] Export functions work
- [ ] UI is responsive

### Automated Testing (if added)
```bash
pytest tests/ -v
pytest tests/ --cov=app  # With coverage
```

## 📚 Documentation

Update documentation when you:
- Add new features
- Change existing behavior
- Fix bugs that affect usage
- Update dependencies

**Documentation includes:**
- README.md
- DEPLOYMENT_GUIDE.md
- Code comments
- Docstrings
- This file (CONTRIBUTING.md)

## 🏷️ Version Control

### Branch Naming
- Feature: `feature/description`
- Bug fix: `fix/description`
- Hotfix: `hotfix/description`
- Documentation: `docs/description`

### Versioning
We use Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

## 🎨 Design Decisions

When making design decisions, consider:
1. **Performance**: Will this slow down processing?
2. **Usability**: Will users understand this?
3. **Accuracy**: Does this improve detection?
4. **Scalability**: Will this work with 1000 videos?
5. **Deployment**: How does this affect deployment?

## 🔐 Security

**If you discover a security vulnerability:**
1. DO NOT open a public issue
2. Email security@yourproject.com
3. Include steps to reproduce
4. Wait for response before disclosure

## 💬 Questions?

- **General questions**: GitHub Discussions
- **Bug reports**: GitHub Issues
- **Feature requests**: GitHub Issues
- **Security**: security@yourproject.com

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

## 📋 Checklist Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added/updated (if applicable)
- [ ] All tests pass
- [ ] PR description is clear

## 🎯 Priority Areas

We especially welcome contributions in:
1. **Performance optimization**: Faster processing
2. **Cloud integration**: S3, Google Cloud Storage
3. **Real-time processing**: Webcam/RTSP support
4. **Mobile app**: iOS/Android
5. **Testing**: Unit tests, integration tests
6. **Documentation**: Tutorials, examples
7. **Accessibility**: UI improvements

## 📅 Release Cycle

- **Minor releases**: Monthly (new features)
- **Patch releases**: As needed (bug fixes)
- **Major releases**: Quarterly (breaking changes)

## 🙏 Thank You!

Your contributions make this project better and help reunite missing persons with their families. Every contribution matters, whether it's:
- Code
- Documentation
- Bug reports
- Feature ideas
- Spreading the word

Together, we can make a difference! 🌟

---

**Questions?** Feel free to ask in GitHub Discussions or open an issue.
