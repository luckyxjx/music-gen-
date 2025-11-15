# 🎵 EMOPIA Music Generator - Running Now!

## ✅ Current Status

Both servers are running and ready to use!

### Backend API Server
- **URL**: http://localhost:5001
- **Status**: ✓ Running
- **Device**: MPS (Apple Silicon GPU)
- **Process ID**: 3

### Frontend Application
- **URL**: http://localhost:5174
- **Status**: ✓ Running
- **Process ID**: 4

## 🚀 How to Use

1. **Open your browser** and go to:
   ```
   http://localhost:5174
   ```

2. **Click "GET STARTED"** on the landing page

3. **Enter a prompt** in the chat interface, for example:
   - "I'm happy, give me an upbeat 2-minute track"
   - "Create a sad melody for 1 minute"
   - "Generate calm music for meditation, 3 minutes"

4. **Wait for generation** (15-30 seconds)

5. **Download the MIDI file** and play it in your favorite music software

## 📊 Available Emotions

- **joy** - Happy, upbeat, energetic
- **sadness** - Sad, melancholic, slow
- **anger** - Intense, aggressive, fast
- **calm** - Peaceful, relaxed, serene
- **surprise** - Unexpected, varied
- **fear** - Tense, anxious, uncertain

## 🔧 Managing the Servers

### View Backend Logs
The backend is running in the background. Check the Kiro terminal panel to see logs.

### View Frontend Logs
The frontend is also running in the background. Check the Kiro terminal panel for any errors.

### Stop the Servers
To stop the servers, you can:
1. Use the Kiro terminal panel to stop the processes
2. Or run: `pkill -f "python3 api.py"` and `pkill -f "npm run dev"`

### Restart the Servers
If you need to restart:

**Backend:**
```bash
python3 api.py
```

**Frontend:**
```bash
cd client
npm run dev
```

## 🧪 Test the API

You can test the API directly using curl:

```bash
# Health check
curl http://localhost:5001/api/health

# List emotions
curl http://localhost:5001/api/emotions

# Generate music
curl -X POST http://localhost:5001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "happy upbeat 2 minute track", "temperature": 1.0, "top_k": 20}'
```

Or use the test script:
```bash
python3 test_api.py
```

## 📁 Generated Files

MIDI files are saved in:
```
./generated_api/
```

Each file is named with a unique UUID (e.g., `a1b2c3d4-e5f6-7890-abcd-ef1234567890.mid`)

## ⚠️ Important Notes

1. **Port Changes**: We're using port 5001 for the backend (instead of 5000) because port 5000 is used by macOS AirPlay Receiver.

2. **Generation Time**: Music generation takes 15-30 seconds on CPU/MPS. This is normal.

3. **Model State**: The model is initialized but not trained. For better results, train the model first using the training guide.

4. **Browser Compatibility**: Works best in Chrome, Firefox, Safari, or Edge (latest versions).

## 🎨 UI Features

- **3D Animated Sphere**: Interactive visualization on landing page
- **Color Cycling**: Sphere changes colors (blue → green → white)
- **Cursor Interaction**: Sphere responds to mouse movement
- **Loading States**: Visual feedback during generation
- **Error Handling**: Clear error messages if something goes wrong

## 🐛 Troubleshooting

### Backend not responding
- Check if the process is still running
- Look for errors in the terminal output
- Restart with: `python3 api.py`

### Frontend not loading
- Clear browser cache
- Check if port 5174 is accessible
- Restart with: `cd client && npm run dev`

### CORS errors
- Make sure both servers are running
- Check that API_BASE_URL in App.tsx matches backend port (5001)

### Generation fails
- Check backend logs for errors
- Ensure PyTorch is installed correctly
- Verify model files exist

## 📚 Next Steps

1. **Train the Model**: See `TRAINING_GUIDE.md` for instructions
2. **Customize Emotions**: Edit emotion mappings in `api.py`
3. **Add Features**: Check `client/INTEGRATION_GUIDE.md` for frontend development
4. **Deploy**: See `FULLSTACK_SETUP.md` for production deployment

## 🎉 Enjoy!

Your EMOPIA music generator is now running. Start creating music with AI!

---

**Need help?** Check the documentation:
- `FULLSTACK_SETUP.md` - Complete setup guide
- `API_INTEGRATION.md` - API documentation
- `client/INTEGRATION_GUIDE.md` - Frontend guide
- `TRAINING_GUIDE.md` - Model training
